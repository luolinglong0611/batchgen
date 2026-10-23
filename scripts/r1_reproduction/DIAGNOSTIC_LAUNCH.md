# R1 双机加载与诊断启动（尚未执行）

本入口用于实际完整权重加载、eager 正确性诊断；不代表 attention 已完成资格验证，也不能用于宣称论文加速比。当前 `triton_precision` 仍由独立完整层测试验证。先确保两个节点部署相同、经审查的源码，以及完成的 163 个转换分片，再由实验负责人启动。不要与单层测试、675 GiB 注册测试或基线服务争用资源。

默认只打印计划，不导入 torch、不建 CUDA context、不创建文件：

```bash
ROOT=/home/admin/batchgen-r1-repro-20261023
cd "$ROOT/repo"
"$ROOT/env/bin/python" scripts/r1_reproduction/launch_r1_diagnostic.py --node-rank 0 --run-id r1-diagnostic-v1
```

节点 1 使用 `--node-rank 1`。实际执行在各自已认证容器内追加 `--execute`，先节点 0 后节点 1；两边尽快启动，分别把 stdout/stderr 保存到新日志。每个节点执行前逐分片重读约 641 GiB 并核对 SHA256，耗时计入加载准备，不计入推理性能。不要在这些检查完成前发送请求。

```bash
# node0 容器；node1 同命令改为 --node-rank 1 及新的本地日志文件名
"$ROOT/env/bin/python" scripts/r1_reproduction/launch_r1_diagnostic.py \
  --node-rank 0 --run-id r1-diagnostic-v1 --execute \
  > "$ROOT/r1-diagnostic-v1-node0.log" 2>&1
```

同 run-id 的节点结果目录已存在时拒绝重用；失败应保留证据并用 v2 等新名字重试。脚本转发 SIGINT/SIGTERM 给自己启动的进程组，等待 30 秒后仅清理该组；不杀其他任务。它不安装依赖、不设置 sysctl、不改 hugepages、不清 GPU/缓存。正常 HTTP 启动入口仍执行真实 runtime preflight，未跳过版本、native 来源、AOT freshness、tokenizer 校验。

## 路径和参数依据

- **使用 `batchgen.launch_http_server`，不是旧 `batchgen.launch_server` 骨架。** `WorkerManager._load_model_resources` → `_load_model_locally` 原样传入 `--converted-ckpt-dir "$ROOT/converted"`。`DeepSeek_Parameter_Server.Init` 非空目录直接 native 加载。无需重转或移动文件。
- 独立 generic `batchgen.parameter_server` 则会追加 `deepseek-ai/DeepSeek-R1`；它不是本脚本启动路径。脚本拒绝非空 `BATCHGEN_PARAMETER_SERVER_ENDPOINT`，防止切换到不同目录语义/权重 owner 生命周期。
- `--model deepseek-ai/DeepSeek-R1` 保留真实注册表 config、675 GiB reservation 与原生 tokenizer。`--cache-dir` 是已下载原版 FP8 目录；不使用 `--model 本地路径`，不再量化，不将模型整体转换 BF16。转换 receipt 要求 `marlin=false`、固定原始 revision、当前审查 converter SHA 和完整输出 hash。
- `world-size=16, nnodes=2`，每节点 8 worker。BasePlanner 默认 EP，双机 attention mode 3，256/16=16 个 routed experts/层/rank，DP attention 与 replicated shared experts。无需额外 EP flag；`enable-ep-with-offloading` 是另一项部分专家 offload 实验，此轮关闭。推理仍有 prefill/decode 阶段权重和 KV 的迁移。
- `host-kv-cache-size=64` 是每节点共享的 **64 GiB**，源码乘 `1024**3`，不应再乘 8。KV BF16、CUDA graphs 关闭、prefix cache 关闭、DeepEP 关闭；原生 NCCL EP 路径。输入 token pool 显式 1 GiB（默认 50 GiB）；保留 prepack 与正常阶段调度。GPU memory frac 0.85 是诊断值，未调优。
- NCCL 使用已验证 eth0 / RoCE GID3 / 8 个 mlx5_bond HCA。脚本继承并记录外部 UCX 环境，不擅自添加未经验证的 UCX transport；正式执行前应采用实际 KV migration 已通过的 UCX 配置。
- HTTP 每节点 10900；协调地址 `10.66.9.163:34155`，restart control `:34156`，PyNccl 34200–34299。端口有冲突时正常参数校验失败，不自动杀服务/改端口。客户端只访问节点0 `http://10.66.9.163:10900`。

节点0需要部署最新 **clean** R1 runtime_preflight 版本并复核：不要复制本地含未提交 Kimi 用户改动的整份文件。相同要求适用于其他有用户改动的源码；用受审查 commit/单独 patch。脚本记录启动时所有 batchgen Python 源文件 hash 和 git commit，但记录不是自动证明两节点一致，执行前应比较。AOT `.so` 由正常 preflight 校验。

## API 与 tokenizer

`GET /health` 和 `GET /v1/workers/status` 确认就绪。当前 HTTP 源码以 `/v1/files` 上传 JSONL、`/v1/batches` 创建任务并轮询/下载结果；`/v1/inference` 返回 410，没有直接 `/v1/chat/completions` 服务路由。JSONL 内部 `url` 可为 `/v1/completions` 或 `/v1/chat/completions`。

原生 `DeepSeekV3Tokenizer` 从包内 tokenizer.json/tokenizer_config.json 读取，非 cache-dir 自动覆盖；BOS=0、EOS/PAD=1。已做的 R1 BOS 配置修正和原始资产必须一并部署，模板/编码与基线需通过 workload manifest 固定。

完整加载后的第一步应是短请求（例如 32–128 输入 tokens、3–16 输出 tokens），不是 6K 长请求性能跑。之后才使用已准备的 `batch_benchmark_client.py --count 1 --warmup-count 0` 对同一 smoke manifest 测长请求。保留实际输出长度、EOS 与错误记录；不能将高精度诊断 attention 的耗时称为 BatchGen 论文性能。

## 内存和仍需实测的风险

每节点权重 owner 的 native memfd reservation 是 **675 GiB**；8 worker 映射同一份物理页，不是 8 份675 GiB，但每个 CUDA context 都要注册整个区域，存在注册延迟、driver页表、锁页/映射资源风险。此前单分片/单worker DMA passed 不证明全区/8worker可用；新增675GiB/8worker测试只填首分片，仍不等于完整163分片加载或叠加KV后的峰值。

静态预算至少 675+64+1=740 GiB，1 TiB名义剩余284 GiB还需供 worker Python/native、页表、metadata、临时 pinned buffers、NCCL/UCX及其他容器进程使用。R1 PS名字解析已使用 meta skeleton，worker空大权重 skeleton已验证，避免另一份671B浮点模型；native加载采用O_DIRECT直接写memfd、非整模型Python中间副本，但8worker初始化与运行时仍可能分配暂存。逐分片hash会产生可回收文件缓存；不可把cache峰值直接等同不可回收anon内存。

不要按8worker RSS简单相加推断真实物理占用（共享页重复计入）；应同时记录容器cgroup memory.current/peak/events、anon/file/shmem、进程PSS、可用内存、每卡显存和registration耗时。GPU263–269GiB并不消除CPU675GiB常驻要求。hostKV同样anonymous memfd，不需要改 `/dev/shm` 挂载。BasePlanner仍有96GiB/H20启发式常数，双机专家布局固定但batch容量未针对B300调优；性能实验前需另审查。

## 已完成的 16 卡 routed-MoE 检查

2026-10-23，`moe_ep_qualification.py` 在两节点16卡通过5个场景共80个rank-case：真实7168/2048维、FP8/FP32 scales、256选8、每rank16个不同synthetic experts；均匀/不均匀tokens、空rank、全部空、全部16个expert-owner ranks与跨节点路由均覆盖。直接执行真实 `DeepseekV3MoE_Decoding_FP8.moe_infer_allgather_allreduce_bf16_acc`，使用原生router/dispatch、两阶段TMA、scatter和PyNccl，不使用通信stub。

固定既有MoE composition gate未调整：最大relative L2 **0.003106992**（阈值0.02），最大abs/reference RMS **0.02835883**（0.15），最小cosine **0.999995174**（0.9998）；router专家集合完全一致。独立参考保留原始FP32blockscales和activation FP8量化，并将每rank的BF16 partial以FP32求和，从而不继承被测BF16 reduction误差。Torch与PyNccl均NCCL2.27.7。

测试脚本SHA256 `2907bc9aa09748ab516c2e8f2e920d09d5179b4e6e2c886770ee2ba93c040335`，两节点所有记录的源码hash一致。远端证据各在 `$ROOT/moe-ep-v1/rankNN.json`、同名 `.pt`、`supervisor.json` 与 `$ROOT/moe-ep-v1.log`；两个supervisor和16workers退出码均0，GPU进程列表再次确认空。每卡测试peak allocated为811243008bytes。该检查不覆盖shared expert、PSM加载、host DMA、attention或完整61层，也不提供推理性能结论。
