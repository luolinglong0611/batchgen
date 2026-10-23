# MGN CUB 修复的二进制部署

`deploy_mgn_cub_fix.py` 默认只打印计划。它为已通过隔离 sanitizer 对照的
`17b9fcbc2a4f9cfb9a5d31c6cb8b7db99f6331bd` 候选准备部署：wheel SHA256
`7df577f7d4cb0e0944806fb0e38275b92f8626589e167729bb39321632bfb015`，
MGN `.so` SHA256 `0f535cbd3c3fda1f4678a4bdec316d083f6383c3ed7fefb1777b935c9f075b2a`。
工具不安装这个 MGN-only wheel，只读取其中唯一的 MGN 扩展。

两台容器的只读新进程检查均确认了两个实际加载路径：

- `repo/batchgen_kernels/common/_C_mgn_ops.cpython-311-x86_64-linux-gnu.so`
- `env/lib/python3.11/site-packages/batchgen_kernels/common/_C_mgn_ops.cpython-311-x86_64-linux-gnu.so`

路径均相对于 `/home/admin/batchgen-r1-repro-20261023`。当前两处旧 SHA256 均为
`ff1ed9bff0688373c02a26c23242a7ec6c1534a49b3e19d9cd6360d2d1359fd6`。
`_get_mgn()` 通过包路径导入 AOT 模块；`BATCHGEN_KERNELS_DEV=1` 允许导入失败后的
JIT。部署探针用 `DEV=0`，分别验证 repo 优先和仅环境包两个新进程，断言精确包、
wrapper、扩展路径及 SHA。探针隐藏 CUDA，断言 CUDA 未初始化，不查询设备或执行内核。

在已核实的容器内执行，`--node-rank` 对应 0 或 1：

```bash
python3 repo/scripts/r1_reproduction/deploy_mgn_cub_fix.py --node-rank 0
python3 repo/scripts/r1_reproduction/deploy_mgn_cub_fix.py --node-rank 0 --inspect
```

后续明确执行时需将同一 wheel 放到该节点的候选路径（或提供 `--candidate-wheel`），
并保持所有 BatchGen 启动器停止。SGLang 可继续运行，工具不会查询 GPU。

```bash
python3 repo/scripts/r1_reproduction/deploy_mgn_cub_fix.py --node-rank 0 --apply --confirm-no-active-batchgen --output /home/admin/batchgen-r1-repro-20261023/mgn-deployment-v1
```

执行前检查固定容器 hostname、旧库 SHA、loader/wrapper SHA、已验证 core SHA。
它持有实际 BatchGen `host.lock` 与部署锁，并在每次替换前扫描 `/proc` 中的
我方 real/effective UID 下的 BatchGen 命令、checkout multiprocessing 进程及
映射 MGN/core 的进程。我方 UID 下无法读取的活进程会导致拒绝。两容器的我方 UID
为 500，实际存在其他 UID（0、1000）的平台进程且 maps 读取返回 EACCES；工具仅
记录这些进程的 PID、UID 和访问限制，不检查命令、不声称整个 PID namespace
无 BatchGen，报告 `uninspected_other_uid` 及 `namespace_complete=false`。
启动前 preflight 可能早于 admission lock，故父任务
仍须保持 BatchGen 启动器停止；进程扫描不能代替该操作约束。

工具先创建独立备份与 `transaction.json`，fsync 文件、事务目录和父目录，再逐个
用同目录临时文件及 `os.replace` 替换两处 `.so`。每个替换原子，但两个位置不是
一个原子操作。其余所有 native 扩展（包括 node0 的旧 `build/lib` 副本）、wrapper
和 `core_engine` 的完整路径/SHA 清单在执行后必须一致。

导入失败、SIGTERM/SIGINT 或替换错误触发原 SHA 备份恢复及原路径 CPU 导入验证。
SIGKILL/掉电后可从持久事务恢复；显式恢复中断同样可以重试原事务：

```bash
python3 repo/scripts/r1_reproduction/deploy_mgn_cub_fix.py --node-rank 0 --rollback-from /home/admin/batchgen-r1-repro-20261023/mgn-deployment-v1 --confirm-no-active-batchgen --output /home/admin/batchgen-r1-repro-20261023/mgn-rollback-v1
```

恢复仅接受原备份 SHA，且目标当前必须为原版或候选 SHA，拒绝覆盖后续未知修改。
工具不更改 `dist-info/RECORD`、包版本或 CUDA 源码；这两处 `.so` 的部署 receipt
才是当前二进制来源证明。后续重新安装包或 `build_ext --inplace` 可能覆盖修复，
启动前须再次校验 SHA。部署成功只说明二进制及 CPU 导入合格，完整模型质量、
greedy 输出一致性和性能仍需独立验证。
