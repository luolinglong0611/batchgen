# BatchGen 三项性能修复与八卡短测

日期：2026-10-28。硬件：同一台 8×NVIDIA L20D，275040 MiB/卡，实测 CUDA capability 为 SM103。

三项修复已接入并通过整模型短测。在双方均启用新 prefill 和原生 FP8 后端的条件下，开启页边界轮换后，32 条请求的热态完成时间从 **127.04 秒降至 74.07 秒，下降 41.7%**。这是同一 BatchGen 版本的调度对照，不能作为相对 SGLang 的加速比。

## 1. 修改内容

| 问题 | 本次修改 | 实际验证 |
|---|---|---|
| Prefill 使用 EP1、从 Host 反复加载专家 | 新增 EP8 常驻 prefill，每卡32个专家；使用 DeepEP normal FP8 和 DeepGEMM contiguous；prefill/decode 共享同一份专家权重 bank | 每卡58层 bank 全程各构建一次；后续 phase 未重建；prefill 的 attention/routed/shared 权重复制队列均为空 |
| SM103 兼容路径的 routed MoE 退化为 FP16 MMA2，通信及量化也有额外开销 | 新增 DeepGEMM 原生 FP8 后端；decode 使用 FP8 UE8M0 DeepEP LL、masked GEMM 和融合激活/重量化；接入整模型 CUDA Graph 的权重生命周期和预热接口 | 真实 GPU 算子、Graph 改输入、零专家和八卡整模型通过；运行日志明确选择 `DeepGEMM_masked` 和 FP8 dispatch |
| 部分序列长时间等待，换入后延迟参与计算，尾部出现空算及重复模型配置 | 新增默认关闭的页边界进度平衡；落后进度桶优先，同桶优先保留驻留 KV；按真实页预算装箱；换入完成后在同一边界接入；修复活动集合为空但仍有待换入请求时的提前返回 | q128 对照中，等待者首次运行提前，空产出 forward 减少，decode 模型配置次数由4次降为1次 |

主要代码：[resident_prefill.py](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/models/deepseek/deepseekv3/resident_prefill.py)、[deepgemm_moe.py](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/models/deepseek/deepseekv3/deepgemm_moe.py)、[策略管理器](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py)、[页调度器](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/worker/r1_page_rotation.py)、[worker](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/batchgen_worker.py)、[整模型 Graph](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/batchgen/attention/mla/r1_whole_decode_cuda_graph.py)。

GPU bank 直接从原 Host FP8 权重逐专家构建，没有同时保留旧 GPU 专家副本。每卡58层 bank 合计约 **76.72 GiB**，跨 phase 共享。此数值是 bank 总大小，并非新增的额外76.72 GiB。

## 2. 明确改变的实验条件

- Prefill 从原 EP1 Host 专家流式加载，改为 EP8 常驻专家；默认 prefill token cap 从旧路径的131072/rank改为8192/rank。
- 权重采用 SGLang 的 Blackwell 数值策略：原 FP8＋FP32 block scale → BF16 → UE8M0 FP8 重量化。原 Host 权重保持不变，GPU 权重数值会改变；没有据此宣称双方整模型精度已经对齐。
- 页边界轮换是新增、显式开启的调度策略。这仍然使用 GPU 常驻工作集，未新增同层 Host KV streaming。
- 新路径依赖现有 DeepGEMM 和 SGLang 原生算子。SGLang 0.5.5.post3 的955个 Python 源文件与安装 RECORD 全部一致，未修改 SGLang 源码。

因此，这些结果属于修改后 BatchGen 的验证，不是未经修改 main 的纯参数复现。Prefill EP8 和原生 FP8 是通用推理优化，不能把它们的收益全部归因于 Host KV 机制。

## 3. 本次短测的实际参数

两组使用完全相同的代码、权重、硬件、请求和后端，仅轮换参数不同。

| 参数 | q0 对照 | q128 修复组 |
|---|---|---|
| 推理任务 | 32条，输入192、固定输出256，忽略EOS | 相同 |
| Attention / MoE 拓扑 | DP8 / EP8，每卡32专家 | 相同 |
| Prefill | 常驻专家、DeepEP normal FP8、cap8192 token/rank | 相同 |
| Decode | 原生 FP8 DeepGEMM、DeepEP LL、整模型 CUDA Graph | 相同 |
| Attention 子批 | 每卡1条 | 相同 |
| Admission / LL capacity | 16/rank，接收 stride128 | 相同 |
| GPU KV | 固定640 token/rank，即10个64-token页；真实 running 为1–2条/rank | 相同 |
| Host KV | 整节点共享2 GiB；静态 chunk256 token | 相同 |
| 初始 / 扩展 / 决策间隔 | 1页 / 1页 / 1页 | 相同 |
| 上下文上限 | 16384 | 相同 |
| 页轮换 quantum | 0，保留旧调度 | 128 token |
| 测量 | 同进程先1次冷态，再1次热态 | 相同 |

**640-token GPU 池仅用于快速制造 Host 库存超过 GPU 容量的机制测试。它不是后续正式性能测试的推荐容量，也不是85% GPU KV压力测量。** 本次未运行6000请求或新的SGLang性能测评。

## 4. 实测结果和收益来源

| 指标 | q0 | q128 |
|---|---:|---:|
| 热态全部请求完成时间，秒 | 127.038 | **74.067** |
| 热态端到端输出吞吐，token/s，整机8卡 | 64.485 | **110.602** |
| 冷态全部请求完成时间，秒 | 178.223 | 106.722 |
| 每批成功 / 错误 / 输出 token | 32 / 0 / 8192 | 32 / 0 / 8192 |
| 每批实际全局 forward 次数 | 892 | **831** |
| 每rank没有新输出的 forward | 126 | **2** |
| 等待序列首次参与 decode 的 forward 序号 | 447 | **128** |
| 等待序列均在首组完成前开始运行 | 否 | **是** |
| 每批 decode 模型配置次数 | 4 | **1** |
| 每次全局 forward 的有用输出行数 | 9.148 | **9.819** |
| 每次全局 forward 的真实 MoE 输入行数，排除通信 padding | 10.278 | 9.839 |
| 每批 H2D 序列载入次数，全部8卡、含初载 | 48 | 64 |

两种策略的冷、热批次各自具有相同的 forward 数、调度轨迹和输出计数。四批合计128条请求、32768个输出 token，错误数为0。

吞吐口径为成功输出token数除以全部请求完成时间：本轮修复组为 `8192 / 74.067 = 110.60 TPS`。此前约1521.49 TPS的测试使用1968条8K→2K请求、每卡约1915648-token GPU KV池，实际running可达187–228条/rank；本轮是32条192→256、640-token小池、1–2条/rank。两种工况不能直接比较高低。本轮同条件对照的吞吐提升为约71.5%；原8K→2K大池工况修复后的TPS尚未测量。

**调度收益主要来自减少空算和重复配置。** q128 的有用输出行数提高约7.3%，但真实 MoE 输入行数略低；这组结果没有证明每次 MoE 的总输入 batch 变得更大。H2D 载入次数增加了，仍然获得更短的任务完成时间。

热态 prefill 的最慢rank KV-ready时间分别为0.542秒和0.619秒。两组均使用新常驻路径，未单独做旧EP1路径的同条件消融，因此这里不给出prefill修复的独立加速比。

计时包括上传、创建批次、轮询和获取全部输出；不包括服务进程启动。Observer及诊断仍有开销，各策略只有1次热态测量。两次冷态运行之间共享磁盘编译缓存，因此不使用冷态差值推导严格加速比。

## 5. 正确性与机制证据

- **CPU：64项相关检查通过**，含57项模型/调度/Graph检查和7项私有observer检查。
- **单卡原生 MoE：** 实际 E32/H7168/I2048、stride128；S1/S3对独立FP32 GEMM参考的最大NRMSE分别为 `5.63e-5`、`3.21e-5`；三组更换输入/有效行数的Graph与eager有效输出逐位一致；零专家通过。两次GEMM及融合激活的热态算子计时均值为0.271ms，仅代表该算子测试。
- **八卡 normal 路由：** 每rank四轮，每rank至少一轮零输入，全部与独立CPU路由参考逐位一致；验证权重只应用一次。
- **整模型 Graph：** 八个rank均观察到完整61层attention、58层真实LL dispatch/combine集合；直接设备计数验证真实replay；首次非零输入的eager/Graph诊断通过。
- **Host机制：** q128已确认的未完成Host KV库存最高达到GPU KV总容量的约2.097倍；观察到不同请求复用同一GPU物理页，ON_HOLD恢复时上下文/生成进度连续，61层Host append没有观测到缺口。
- **层内合批：** 真实attention子批 `[1,1]` 合并后进入2条本地有效行的MoE；LL capacity16的padding未计入有效batch。
- **清理：** 两个服务均正常退出0，没有强制清理，最终Host/GPU页全部归还；最后检查8卡均仅约4MiB背景占用，没有计算进程。

原生DeepGEMM可能写满一个M tile，partial tile的padding不是有效输出；数值比较和完成token计数均以有效行数为准。

## 6. 启用方式与适用范围

在现有SM103环境及R1八卡启动参数中显式设置：

```bash
export BATCHGEN_DEEPSEEK_SINGLE_NODE_RESIDENT_EP=1
export BATCHGEN_R1_MOE_BACKEND=deep_gemm
export BATCHGEN_R1_RESIDENT_EP_PREFILL=1
export BATCHGEN_R1_PREFILL_TOKENS_PER_RANK=8192
export BATCHGEN_R1_PAGE_ROTATION_TOKENS=128
export BATCHGEN_R1_DECODE_CUDA_GRAPH=1
export BATCHGEN_R1_WHOLE_DECODE_CUDA_GRAPH=1
# 启动器同时传入 --enable-deepep --enable-cuda-graph
```

旧后端及旧调度仍是未设置这些opt-in时的默认行为。正式实验应重新使用大GPU KV池并调优并发；capacity448/stride3584、满KV及原8K→2K/6000任务尚未获得本次新路径的实测资格。当前实现要求权重不可热更新。

## 7. 记录与复核

本轮基于已冻结的 `39ba5dfa877051fc7db87fd23b376e1c431ab1b3` 和前序source-v5适配继续开发。两组远端运行目录分别为 `campaigns/r1-perf-fix-short-q0-20261028-v1`、`campaigns/r1-perf-fix-short-q128-20261028-v1`。共同源码清单SHA256为 `97bc3a48434e4f741561af3a24ed062c44cd5cad793107d380eeddf14f895189`。

证据：[q0分析](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/q0/short-test-analysis-v2.json)、[q128分析](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/q128/short-test-analysis-v2.json)、[阶段配置日志摘录](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/phase-ledger.json)、[原生MoE GPU结果](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/native-moe-gpu.json)、[八卡路由结果](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/normal-prefill-routing-gpu.json)、[退出和SGLang源码复核](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/reports/r1-single8-20261028/performance-fixes/final-settlement.json)。各组目录同时保存完整请求计数receipt和机制分析。

可复用GPU短测脚本：[原生MoE](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/scripts/r1_reproduction/smoke_r1_native_moe_gpu.py)、[八卡normal路由](/Users/lilele/.codex/worktrees/r1-main-validation/batchgen/scripts/r1_reproduction/smoke_r1_normal_prefill_gpu.py)，默认仅输出计划，显式 `--execute` 才执行GPU测试。
