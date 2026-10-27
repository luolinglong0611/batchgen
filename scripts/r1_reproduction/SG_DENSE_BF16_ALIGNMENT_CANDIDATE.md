# SG dense BF16必要对齐与孤立层资格

初始实际回执发现BG原main前三层dense为BF16，而SG原FP8。该候选保留BG main，**仅**把SG层0–2的dense改为同一反量化公式及独立gate/up/down运算顺序；原checkpoint、attention/shared/routed experts和BG数学均不变。不是论文原软件条件，必须在实验报告注明SG这一必要参数对齐改动。

默认`SGLANG_R1_DENSE_BF16=0`，不会导入新helper或更改SG模块。显式1才安装，并要求冻结原checkpoint派生的18项dense weight/scale provenance JSON及SHA。严格pin SG模型源码`b37416df…`、FP8源码`31faac…`、BG dequant源码`2872df…`，实际SM103、Graph OFF、明确`model.layers.0..2.mlp`、TP1、无bias、原FP8/FP32scale及原形状；未知布局/源身份/原值差异均拒绝。安装后未调用仍是unverified。

第一次真实MLP调用在原loader及其FP8后处理完成后逐chunk核验原始权重/scale bytes；合并gate_up按18432行拆开，scale按144行拆开，分别与原tensor SHA核验。先准备全部BF16结果，再原子替换该层两份参数，失败回滚。转换调用原BG PyTorch反量化函数：

`W_bf16[i,j] = BF16(FP32(W_fp8[i,j]) * S_fp32[floor(i/128), floor(j/128)])`

每次forward直接执行gate `F.linear` → materialized BF16 `F.silu` → up `F.linear` → materialized BF16乘 → down `F.linear`；不经过FP8 activation quantizer或SG `SiluAndMul`。旧FP8 quant_method被明确拒绝器替代，不能误消费派生BF16权重。只允许已准入的无fusion/scatter/zero-allocator路径。shared仍用原候选的普通FP8路径，不因这个修复宣称其所有融合/舍入/通信与BG相等。

三层永久权重增加1189085184bytes（1.107421875GiB）/GPU。反量化/拼接/byte校验另有临时GPU与CPU开销，未实测不得以永久增量代替真实headroom。lazy转换必须在同一进程预热完成，排除正式BCT，并重新采集actual dtype、math入口和容量。回执`dense_backend_receipt()`为defensive copy，记录已转换层、原tensor证明、实际dtype/shape/stride、公式、host forward API返回次数；计数不等于CUDA同步或数值资格。

## CPU准备原权重provenance

默认只输出plan，无GPU/HTTP/模型加载：

```bash
python3 -m scripts.r1_reproduction.qualify_sglang_r1_dense_bf16 --checkpoint-dir ORIGINAL_R1 --model-manifest r1_model_manifest.json
```

显式CPU准备：父guard已分配主机，使用空GPU可见性和fresh输出目录。

```bash
R1_DENSE_PROVENANCE_CPU_ONLY=1 CUDA_VISIBLE_DEVICES='' python3 -m scripts.r1_reproduction.qualify_sglang_r1_dense_bf16 --checkpoint-dir ORIGINAL_R1 --model-manifest r1_model_manifest.json --expected-hostname ALLOCATED_HOST --output FRESH_PROVENANCE_DIR --prepare-provenance
```

manifest必须SHA为`d9b7aeac6a69ad2f40d885f04bd7c1eaeac766f681e6be374980edf8c504a253`。工具用冻结index寻址，逐个相关原safetensors文件核验**整文件**SHA/size，再独立解析headers/offset并hash18份真实tensor bytes；不依赖陈旧index.total_size，不转换/写原文件。完整model identity引用已经独立seal的身份，本工具不重新宣称全91991tensor资格。

## 单GPU实际层资格（v2已通过）

父guard应只暴露一张SM103、限制8GiB Torch allocation，并用明确hostname/fresh输出、当前worker/helper/runtime SHA绑定执行。环境必须逐项显式设置：

```text
R1_DENSE_BF16_QUALIFICATION_GUARDED=1
SGLANG_R1_DENSE_BF16=1
BATCHGEN_R1_SM103_FP8_COMPAT=1
DISABLE_MMA_V5=1
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_ENABLE_FLASHINFER_FP8_GEMM=0
SGLANG_R1_COMMON_FI_FA2=0
R1_FULL_MODEL_TRACE=0
```

```bash
python3 -m scripts.r1_reproduction.qualify_sglang_r1_dense_bf16 --checkpoint-dir ORIGINAL_R1 --model-manifest r1_model_manifest.json --expected-hostname ALLOCATED_HOST --provenance FRESH_PROVENANCE_DIR/dense-provenance.json --provenance-sha256 PROVENANCE_SHA --worker-source-sha256 WORKER_SHA --helper-source-sha256 HELPER_SHA --runtime-source-sha256 RUNTIME_SHA --output FRESH_GPU_DIR --execute
```

该worker真实构造SG原生TP1 `DeepseekV2MLP`，复制原FP8/scale并经过native FP8后处理，调用**实际安装后的同一lazy helper forward**。三层依次测试M=0/1/2/4/16/23/64/128；每个派生BF16权重值与独立CPU FP32乘scale→BF16公式比较，forward与BG main真实`DeepseekV3MLP.forward`在同原权重上要求BF16字节完全一致、全部有限，失败不调阈值。原始18tensor值、相关文件及执行源码在运行前后校验。

这证明的是孤立dense必要对齐契约。它不验证full SG checkpoint loader、16rank布局、全部正式prefill形状、FP64完整MLP精度、整模型数值/语义或性能，成功时仍`formal_test_permitted=false`。v2已现场通过以下限定契约；CPU测试和孤立GPU通过均不冒充整模型通过。

现场CPU进展（协调者回报，非GPU资格）：node0已生成18项原dense tensor proof，provenance SHA`67aa80b4274b0d61dd2f58b5ec0829120c13de9b4aae319d8280b27e8a47dc2a`，10秒CPU monitor正常退出且cleanup完成。本地本次没有取得/复算完整proof值，不把协调者回执改写为本地原bytes复核；GPUworker仍会重验原文件和tensor证明。worker最终SHA`81731a563df153b17957d82bb9f4cf5ffa79502484bfb5f374f5d562509c94ac`，新增依赖source posthash检查；无需重跑已完成CPU准备。GPU/整模型/性能仍未准入。

## SG dense v2真实孤立GPU通过（更新）

`SG_DENSE_BF16_RESULTS_20261026.json`保存已取回的完整原始result bytes，SHA`676fd2878dd349bcd17129e115239d2744bfa35766f4758a241e7566edc10839`本地复算一致。node0单张SM103实际原生SG TP1 MLP，通过生产同一lazy helper与BG main真实MLP逐一比较：3层×8种M（0/1/2/4/16/23/64/128）全部BF16字节相同、有限且maxabs0；9份完整派生权重、共1189085184个BF16元素与独立CPU反量化公式一致，18项原FP8/FP32scale实际bytes证明匹配。Torch peak_allocated=4104547840bytes，8GiB分配上限内；不是整GPU/NVML峰值，也不是完整模型峰值。

`repair-status3-node0.json`记录退出0、cleanup.complete=true、sources_unchanged=true。v1的`flash_attn_interface`缺失导入失败（failed SHA17376b9324133d998bdfdd4fe9dc53f0635ff46c41cb985f0694275b0f861fac）保留：失败发生于导入BG model、尚未执行dense数值case。v2新增显式`BATCHGEN_R1_SM103_MLA_COMPAT=flashinfer-fa2`解决该导入路径，另换run/output/cache身份；worker81731a…、helper097110…、runtime424e62…、数学和阈值均未更改。

此通过只覆盖孤立dense必要对齐契约：原tensor复制到真实SG TP1层，真实full SG checkpoint loader、16rank整模型、全部正式长prefill形状、FP64完整MLP精度和语义均未因此通过。旧SGv3/BGv2回执及其数值／语义硬失败仍保留。完整模型资格与正式测试仍false；RoPE v1/v2当前1 BF16 ULP数值检查失败，最终修复及新source实测通过前，新SGv5/BGv3 generator不得release。
