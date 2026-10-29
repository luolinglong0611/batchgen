# R1 SM103 6000 请求性能版本冻结

冻结标签：`r1-sm103-priority-opt-20261029`；目标分支：`lll`；运行代码提交：`342f7c002fe49fcabb109f73a24e8c41956c5d1d`。
标签所在的最终提交只添加冻结材料和验证工具；运行源码与已完成的 6000 请求测试相同。
本次没有改变运行源码来整理提交。

## 功能提交

| 提交 | 功能 | 改动 |
|---|---|---|
| `9f43061` | 压缩分词通信 | 以小型元数据加紧凑 token tensor 交换，避免大量 Python token 对象序列化 |
| `9165b25` | 共享输入量化复用 | Q/KV 投影、共享专家 up/gate 在同次 forward 复用同输入的 FP8 量化结果 |
| `44c2912` | Host KV 快照 | 每次整模型 Graph replay 后一次性生成连续稳定快照，保留异步追加的存储生命周期 |
| `1b5db08` | Prefill attention | 接通 Blackwell CUTLASS varlen FMHA，并复用计划与 workspace |
| `ddc6bea` | MoE 通信配置 | 单独设置 normal prefill dispatch/combine 的 SM 和缓冲区，不修改 decode LL 全局配置 |
| `cc8a9ac` | MoE 激活量化融合 | 融合 SwiGLU、BF16 舍入、per-128 FP8/UE8M0 量化 |
| `3807047` | Decode RoPE 融合 | strict FP32 原生旋转与 latent RMSNorm 融合，去掉重复计算与 Q clone |
| `342f7c0` | 静态类型转换复用 | 缓存静态 router/rotary 转换，加上 Graph 签名失效与存储生命周期守卫 |

前三项是上一轮 5085.99 TPS 版本中尚未固化的改动，其余五项是本轮优先级优化。
完整 SHA 和文件归属见 [SOURCE_FREEZE.json](SOURCE_FREEZE.json)。

## 三项重点优化

### Prefill attention

在原 FlashInfer 0.5.2 中接通 Blackwell CUTLASS FMHA，保留原 BF16 Q/K head192、V head128、causal mask 和模型 YaRN scale。
本路径仅接收通过几何和边界检查的 fresh prefill；CPU 长度元数据避免逐层回读，每个线程/rank 在同一 CUDA stream 上复用一个计划与 workspace。
对 prefix、自定义 mask、非对称 Q chunk 等不支持条件显式拒绝；默认仍使用原后端，需设置 `BATCHGEN_R1_PREFILL_ATTENTION_BACKEND=cutlass-sm100`。

8K、128 heads 的原生单层调用为 6.397232 → 2.049760 ms，约 3.12 倍。
对 SDPA/旧 FA2 的算子容差检查通过，最大绝对差 0.015625；该后端不是逐字节等价。
共享的计划资源约为 32 MiB workspace + 1.50 MiB 元数据/rank；decode attention 路径继续使用原 FA2。

### MoE 通信调优

为 normal prefill DeepEP dispatch/combine 分别传入局部 Config，从原 20 SM 扫描到 40/80/120/144 SM。
最终选 80 SM：单层链路最大 rank 中位数从 9.308800 降到 6.227904 ms。
它比 144 SM 慢约 0.90%，但每卡少用约 1.00 GiB NVL buffer。

80 SM 的 NVL buffer 约 1.251 GiB/卡，较原 20 SM 多约 0.938 GiB。
保留 EP8 chunk 参数与路由/专家计算；8 rank 真层输出在扫描中逐字节相同。
不调用全局 `Buffer.set_num_sms`，decode low-latency 路径仍保留原全局 20 SM。
配置是 `BATCHGEN_R1_PREFILL_NORMAL_SMS=80`、`BATCHGEN_R1_PREFILL_DISPATCH_SMS=80`、`BATCHGEN_R1_PREFILL_COMBINE_SMS=80`。
零值保留安装版本默认设置。

### 量化融合与复用

原来 FC1 后先单独执行 `SiLU(gate) * up`、写 BF16 中间张量，再启动 per-128 FP8 量化供 FC2 使用。
现在用一个 Triton kernel 完成 SwiGLU、BF16 舍入、amax/UE8M0 scale 打包和 FP8 写出，省去中间张量的显存往返与一次 kernel launch。
BF16 舍入边界仍在寄存器中保留，scale floor 仍为 `max(amax/448, 1e-10)`；top-k 权重位置保持原实现。
通过 `BATCHGEN_R1_PREFILL_FUSED_SILU_QUANT=1` 启用，默认关闭。

28 个随机、零/极小值、指数边界和尾部案例的 FP8 值与 packed scale 字节相同。
去掉行数编译特化，合格的六种 M 值共用一个 JIT 实例。
另将同一次 forward 的 Q/KV、共享专家 up/gate 输入量化复用，减少对同一动态输入的重复量化；不会跨 forward 缓存激活。
80 SM 加融合量化的真实 8 rank MoE 链路从 9.309952 降到 5.995104 ms，约 1.55 倍。

## 已完成的测试

| 6000 条 8192→2048 离线测试 | BCT（秒） | 输出 TPS |
|---|---:|---:|
| 原优化 BatchGen | 2416.048796183 | 5085.989993 |
| 本冻结版本 | 2041.614737845 | 6018.765329 |
| 既有 SGLang MegaMoE 基线 | 1696.535748986 | 7242.995031 |

本版本 6000/6000 成功、0 错误；49152000 输入/12288000 输出 token。
相对原 BG，TPS +18.34%、BCT −15.50%；相对 SG，TPS 仍低 16.90%。
Prefill 从 861.4 降至 523.6 秒（−39.22%），decode 从 1514.3 降至 1473.1 秒。
各阶段为服务日志的舍入时间，不能与 BCT 严格相加；Host→GPU READY 等待 63.093044 秒已包含在 decode。

双方使用同一 8×L20D SM103 和相同输入，显存比例均为 0.95；各自调优并发和 KV 池。
本版本 Host KV 1341 GiB、每卡 GPU KV 1968576 token、主批每卡 192 行。
轨迹确认每组 Host 中 2000 条未完成序列，对应 GPU 活动 1536 条、等待 464 条；正式任务有三组。
原有 Host 库存/页面轮换机制、decode CUDA Graph 和 DeepEP LL 继续生效。
256 条完整短测与算子验证已先行通过。

正式 BCT 从首次测量 HTTP 提交前到最后响应字节接收完毕，包含服务分词和 BG 上传/创建/轮询/下载；
同一 PID 的 64 条完整 8K→2K 预热、加载、后验校验和关停全部排除。
正式运行关闭 profiler/decode trace；此次整理提交没有重新启动 GPU 测试。

每个正式版本仅一次有效热态测量；没有逐补丁的全量消融，局部倍数不能直接外推成 TPS。
BG 的 dense BF16、FP32 router、per-128 量化和 top-k 权重位置与 SG 仍有差异；SG 基线含实验性 FP8 MegaMoE 接线及不同依赖版本。
attention 通过的是算子容差检查；没有完成跨框架整模型精度对齐。
输入来自既有 LongBench 派生工作负载，不声称是论文作者的原始请求文件。

## 核验与运行配置

[SOURCE_FREEZE.json](SOURCE_FREEZE.json) 记录 536 个运行源码、4 个 Linux 二进制和 28 个 CPU 测试文件的 SHA256。
[runtime-config.json](runtime-config.json) 保存正式任务的完整 argv、显式环境变量、继承控制项及实际容量；
[validation.json](validation.json) 保存算子/短测/正式结果、比较条件及原始凭证 SHA。
原始日志与输入/输出文件仍保留在现有实验归档中，不随源代码冻结提交。
模型权重、部署依赖和本机编译扩展需另行准备，记录的绝对路径/网卡名需按环境调整；`--storage-path` 应指向新的任务目录。

在仓库根目录验证当前版本：

```bash
python3 scripts/r1_reproduction/verify_frozen_r1_sources.py
python scripts/r1_reproduction/run_r1_cpu_tests.py
```

提交前重新运行的 356 项 CPU 测试全部通过（32.38 秒），564 个运行源码及测试文件的哈希全部一致。
CPU 测试需安装 pytest、CPU PyTorch 和 safetensors；验证脚本自身仅用 Python 标准库。
在原 Linux GPU 环境中可加 `--check-binaries` 检查四个已部署二进制。
不匹配的本机编译产物需要按对应依赖重新资格验证，不能把源码哈希通过当作 GPU 支持证明。

原固定点 `r1-sm103-hotpath-20261028` 及其冻结清单保留；切回原标签后可用验证器的 `--manifest` 指定原清单。
