# R1 双机 B300 内存预算与调参顺序

本记录依据固定代码 `3b5edef4af3710247a5081a4d4847a82aeeb8e1c` 的只读审计，
不是新的容量或性能测试。每节点容器可见约 1 TiB 主机内存、8 张 SM103 GPU；
NVML 名称为 NVIDIA L20D，每卡约 268.6 GiB。完整代码出处和冻结源码 SHA 保存在
实验的 `r1-b300-memory-planner-audit-3b5edef.json` 回执中。

原生参数服务器每节点预留 675 GiB CPU 权重池，8 个 worker 注册同一物理池，
不能把每个 worker 的共享映射 RSS 再累加为八份权重。当前普通加载还包括约
19.3 GiB serialized skeleton，以及每 worker 约 19.3 GiB 的反序列化 CPU skeleton。
这些与 Host KV、token pool、Python/native 临时存储共同占用主机内存。

R1 BF16 MLA KV 每 token 的数据量为 `61 × 576 × 2 = 70272 bytes`。
10240-token 完整轨迹为 160 个 64-token page，约 0.670166 GiB。
下表只有 KV payload 理论容量；实际还需 metadata、临时状态和调度余量。

| 每节点 Host KV | 完整轨迹/节点 | 双节点合计 |
| --- | ---: | ---: |
| 64 GiB | 95 | 190 |
| 128 GiB | 190 | 380 |
| 192 GiB | 286 | 572 |
| 224 GiB | 334 | 668 |

如果同时驻留 1024 条这样的完整轨迹并均分，两节点各需 343.125 GiB Host KV。
`675 + 343.125 = 1018.125 GiB`，尚未计算 skeleton 和运行时，因此不适合直接
配置在 1 TiB 容器中。提交 1024 或 6000 条请求不意味着它们必须同时驻留：
prefill admission 按真实 Host free pages 接纳，其余排队分波执行。
协议中的 `K=1024` 是 token 单位换算，不是运行时 wave 目标。

历史运行观察到约 164.93 GiB GPU KV、39376 pages/rank。这来自真实总显存、
`gpu_memory_frac` 与模型加载后已用显存，独立于 planner 内的 H20 96 GiB 估算。
现代 mode3 decode 采用实际 pages 和运行时 padding cap；不能把 planner 算出的
MoE=44 当成当前硬上限，也不能只把 96 改成 268 就宣称解除了瓶颈。
当前预算还由 rank0 广播，未取 16 rank 最小可用预算；eager 稳态峰值也需实测。

首先保持 Host KV=64 GiB、skeleton mmap OFF，完成 MGN 修复后的功能探针。
之后独立验证默认关闭的 `BATCHGEN_R1_SKELETON_MMAP=1`：测量全模型 PSS/USS、
cgroup 峰值、COW 与输出，并确认 owner 在 worker 打开/重启前保留 memfd。
理想消除 `8 × 19.3 ≈ 154.4 GiB` 的私有存储只是估算；尚无全模型实测节省。
测量通过后才依次试 128、192、224 GiB Host KV，并保留内存余量。

调优时分别控制 prefill token cap、decode rank cap 和 Host KV 容量；记录实际
KV 分配与 planner 预测的区别，再冻结正式实验配置。不要按显存比例直接放大
H20 的 `MAGIC_NUM=672000`，也不要把当前小诊断配置当作最终 SGLang 基线。
