# SGLang R1 双机诊断计划（尚未启动模型）

本计划对应已安装的 `sglang-baseline-v8/baseline-env`，不是重新安装环境。
每节点 8 卡，TP16 / DP-attention16 / EP1，共 16 卡。模型和 tokenizer 均直接读取
`/mnt/fuse/oss/luolinglong.lll/models/DeepSeek-R1-original-fp8`，不读取 BatchGen 转换分片。
诊断仍使用 `full_model_correctness.py` 已固定的 6 个探针、greedy、128 个新 token、
2 次重复、单请求并发；不据此声明质量达标或论文性能复现成功。

## 生成和检查计划

本脚本没有 execute 模式，不会启动模型、初始化 CUDA 或修改第三方源码。
在每台已分配容器中分别运行（rank 为 0 / 1，输出文件须不存在）：

```bash
cd /home/admin/batchgen-r1-repro-20261023/repo
python3 scripts/r1_reproduction/launch_sglang_diagnostic.py \
  --node-rank 0 --inspect --output /home/admin/batchgen-r1-repro-20261023/sglang-node0-plan.json
```

`--inspect` 校验原始 163 分片的当前大小及历史下载 receipt，重新计算 config/index/tokenizer
等小文件 SHA，做首片 4 KiB 的只读 mmap 检查，并将当前关键安装源码 SHA、安装审计 SHA、
实际 CPU-hidden `--help` 和版本记录下来。它不会重新读取 641 GiB 权重计算 SHA，
不会把历史 receipt 冒充当前全部权重字节校验，也不会把所选源码文件校验冒充完整源码树验证。
2026-10-23 已只读核实 node1 的目录、config、安装 help 和 4 KiB mmap 可用；
这不构成全模型加载或 GPU kernel 资格验证。

计划中的节点 0 HTTP 端点为 `http://10.66.9.163:30000`，双机 rendezvous 为
`10.66.9.163:33155`。`shell_command` 是待资源释放及 kernel 检查通过后执行的完整命令。
父任务统一管理启动、日志、7200 秒启动/预热上限和本次进程组清理；不要把本计划当作已经运行。

## 精度和实际后端

- `--dtype bfloat16 --quantization fp8 --kv-cache-dtype auto`：原始 FP8 E4M3 权重、
  FP32 block128 scales；auto 在该模型配置下使用 BF16 KV。TRTLLM MLA 的该版本 CLI
  会拒绝显式 BF16，必须在启动日志看到 `Using KV cache dtype: torch.bfloat16` 后才确认。
- Attention 为 `trtllm_mla`，路由 MoE 为 `flashinfer_trtllm`，a2a 为 none。
  Shared-expert fusion 显式关闭；此 MoE backend 本来就会自动关闭它。
- Dense/shared/attention projection 使用 SGLang 自带 Triton FP8 GEMM，明确设置
  `SGLANG_ENABLE_JIT_DEEPGEMM=0`、`SGLANG_ENABLE_FLASHINFER_FP8_GEMM=0`、
  `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8=0`。这避免 SM103 的 DeepGEMM UE8M0 权重重新量化。
  同时关闭 `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`。
- 不能简单改为 `SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1`：TP16 共享专家 down projection
  的 K=2048/16=128，而 FlashInfer 0.5.2 的 TRTLLM dense GEMM 有 K>=256 的断言。
  `--moe-dense-tp-size 1` 只改变前三个 dense MLP，不能修正此 shared-expert 限制。
- TRTLLM routed-MoE 的 C++ 接口直接接受 FP32 block scales，intermediate=128 满足其
  %128 约束。没有在审查到的 Python/C++ 调用路径看到 UE8M0 转换；仍需真实 GPU 测试。
  MLA 的 kv_b absorption 会按原 FP32 scales 反量化为 BF16；这和运行时 kernel 舍入、
  路由 bias 的 dtype 一起必须记录，不能据原始 checkpoint 相同声称数值逐位相同。
- 关闭 CUDA graph、prefix/radix cache，省略全部 speculative/draft 参数；不启用 torch.compile。
  不继承 BatchGen 的 `DISABLE_MMA_V5` 或 CUDA13 ptxas override，保留 baseline 固定 Triton
  自带的 Blackwell ptxas12.9。必须使用 baseline 的私有 C++ runtime `LD_LIBRARY_PATH`。

GPU 验证优先覆盖 BF16 与普通 Triton、FP8 dense 的 M=1/2、QKV N2112/K7168、
shared-up N256/K7168、shared-down N7168/K128、dense-up N2304/K7168、
dense-down N7168/K1152，再验证 TRTLLM MLA/MoE 的本拓扑路径。
小测试保留输入/原权重/原 scales，分别对比相同激活量化的 FP32 参考和未量化输入参考；
未定义的整体质量门槛不得凭观察结果临时生成。

## 加载内存和计时

保留默认 mmap，并显式令 `enable_multithread_load=false`。不要加
`--weight-loader-disable-mmap`：该选项会 `f.read()` 整片；模型内部还有独立的
ThreadPoolExecutor，为每个 CPU tensor 无界提交 H2D，禁用分片多线程并不等于禁用此队列。
队列积压时，关闭 mmap 可让每个 rank 保留大量私有完整分片，最坏规模超过每节点 1 TiB。
多线程分片 iterator 还会保留所有 futures/results，因此也不启用。

默认 mmap 让 8 个 rank 映射同一组约 641 GiB 文件，页面缓存可共享/回收，模型参数直接
初始化在各自 GPU 上；不会主动建立 BatchGen 那种每节点 675 GiB 常驻 pinned 权重池。
但 FUSE 缓存、H2D 排队、临时转换和首次 JIT 的实际峰值仍需观测，不承诺单片内存硬上界。
必须先释放两台 BatchGen workers **和 parameter-server/host-KV 所有者**；初始建议每节点
MemAvailable>=850 GiB 且 8 卡均空闲，加载期间低于128 GiB 时停止本次启动进程组。
不执行宿主机 drop_caches，不改平台资源，不删除历史实验进程。

每个 rank 的 `--max-total-tokens 32768` 为本轮诊断限制，61×576×2 字节对应约2.145 GiB
MLA KV 数据，另有页表/工作空间。`--mem-fraction-static 0.85` 仍保留。
`--max-running-requests 16`、chunk65536 在 DP16 下被调整为4096/rank。
这些限制服务于短探针，不代表正式吞吐最优配置。

16 个 rank 都会遍历原始163分片。OSS读取/文件缓存和首次编译可能显著影响冷启动时间；
记录加载、JIT、warmup耗时并单独报告。请求诊断/后续性能计时必须在所有rank就绪、
日志确认BF16 KV、kernel检查结束后开始，不能把冷启动和推理吞吐混在一起。

## 源码依据

关键安装文件哈希见 `sglang_diagnostic_source_pins.json`（仅所列文件）。
同版本上游定位：

- `sglang/srt/server_args.py:1253–1269,1378–1414`：TRTLLM MLA KV 限制、DP chunk、MoE配置。
- `sglang/srt/model_executor/model_runner.py:1547–1586`：auto KV dtype 的实际选择。
- `sglang/srt/model_loader/weight_utils.py:610–696`：mmap/eager分片和多线程结果生命周期。
- `sglang/srt/models/deepseek_v2.py:3660–3913`：CPU权重异步提交；3454–3532：UE8M0重新量化。
- `sglang/srt/model_loader/utils.py:123–129`：UE8M0条件；
  `layers/quantization/fp8_utils.py:118–149,295–320`：原FP32-scale Triton选择与调用。
- `flashinfer/gemm.py:2257–2272`：TRTLLM dense K>=256；
  `flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu:624–660`：路由MoE的FP32 scales。

所有启动条件仍须以实际日志和独立GPU记录确认。原有BatchGen严格数值失败记录保持不变。
