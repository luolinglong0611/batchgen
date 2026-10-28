# R1 FP8 → FlashInfer MXFP8 MegaMoE 实验接线

固定 SGLang v0.5.22、提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`。在单机 8×L20D SM103 上，算子、CUDA Graph、完整模型和热态 1440 条 8K→2K 请求已验证。实测 **7474.29 output TPS**，较历史 6497.82 提高 15.03%。结果摘要和原始 receipt 校验值见 [VALIDATION.json](VALIDATION.json)。

这是修改后的 SGLang 实验候选。接入官方 FlashInfer 0.7.1 MXFP8 MegaMoE；不使用 DeepGEMM 的另一套 MegaMoE 接口。原 0.5.5.post3 环境、BatchGen 已冻结代码、磁盘上的 R1 checkpoint 均未修改。

## 修改和部署

`PINS.json` 固定三份候选文件和原始文件的 SHA。`candidate/` 保存实际部署的相同字节，`r1_fp8_megamoe_v0522.patch` 是完整补丁。只新增 routed-weight helper，在 routed post-load、真实 runner 返回日志及 FP8 guard 处接线。dense、attention、shared 的量化配置保持原值。`PINS.json` 的 status 是部署前快照，后续 GPU 资格由报告中的运行 receipt 补充，不重写历史 pin。

在**独立的、字节匹配的**上游源码中检查和应用：

```bash
python3 apply_candidate.py --repo /absolute/sglang-v0522
python3 apply_candidate.py --repo /absolute/sglang-v0522 --execute
```

默认只读；部分应用、未知版本和候选字节不匹配会拒绝。此工具处理源码树，不自动寻找旧安装环境。

必要选项：

```bash
export SGLANG_R1_FP8_MEGAMOE_BRIDGE=1
export SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK=8192
export SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=bf16
export SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=0
# 传给新 SGLang 启动器：
# --quantization fp8 --tp-size 8 --attn-dp-size 8 --dp-size 1 --ep-size 8
# --moe-a2a-backend flashinfer_megamoe
# --moe-runner-backend flashinfer_megamoe --disable-shared-experts-fusion
# --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
```

完整实测 argv、环境及编译兼容设置保存在原始验证归档的 `candidate-service-configuration.json`。本容器需要私有 GCC13、CUDA driver 链接和兼容 CUDA13.0 的构建头文件；不能只复制上述选项到未经准备的旧环境。普通 decode Graph 已验证，未验证 breakable Graph。

## 数值和内存条件

原 routed FP8 E4M3＋FP32 block128 scale 先按旧 C1 的 BF16/UE8M0 FP8 策略处理，再把同一 scale 展开到每行 K32，由官方函数布局转换。逐专家处理，未建立第二份完整模型权重。新 scale 表相对原 FP32 block scale 增加约 42 MiB/层/卡，58层约2.38 GiB/卡；加载和布局转换另有临时内存。

新后端使用 per32 激活量化，top-k 权重在 SwiGLU 后、FC2 前应用。旧 C1 使用 per128 激活、combine 侧应用权重。未声称整模型精度等价；7474 对 6498 是版本和后端一起升级的结果。

## 验证工具

```bash
python -m pytest -q test_bridge_cpu.py
python smoke_r1_mxfp8_megamoe_gpu.py  # 默认只输出计划
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  smoke_r1_mxfp8_megamoe_gpu.py --execute --capacity 8192 --rows 180 \
  --output /absolute/new-operator-result.json
```

CPU 10项检查通过；公开目录导出后再次运行，`10 passed in 0.96s`。GPU 脚本需要本次报告中固定的依赖和整套 reference 文件，使用真实 R1 专家几何、跨八卡路由、零输入 rank、三次改输入 Graph replay；稀疏测试权重用于定位算子问题，不能代替整模型精度评估。

部署日志每层/rank 一次记录准备完成和 native runner 的 eager/capture 成功返回。Python capture 记录不能充当逐次 GPU replay 计数。整模型测试另以原生 `cuda graph: True` 日志及完整任务 receipt 验证。

保留原生按 capacity 的启动调优。源码和实际日志确认只调优10档、跨58层复用 winner；并非每层重新搜索。env8192 是最大 prefill 容量，稳态每rank180条选择180档。启动调优不计入热态 BCT。
