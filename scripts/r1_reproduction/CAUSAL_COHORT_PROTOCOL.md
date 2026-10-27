# 新版因果 cohort 证据协议

状态：CPU 回归已通过，待 root 最终冻结源码 SHA 并部署，在已有 guard 下执行新整模型正确性资格。本文没有执行远端服务或 GPU，整模型与正式性能资格仍待验证。模型数值阈值、答案、请求/jobs 与输出预算保持原契约。

## 新套件与绑定

必须显式选择 `--time-basis node_monotonic_causal_v2 --cohort-size 16`。原套件、原部署/run 身份和旧结果不能改名复用；重新 prepare/import 并冻结新 suite、collector、trace、causal helper、finalizer 及 deployment 的源码/产物 SHA。以下命令省略 `--execute` 时只输出计划；真正执行由 root 冻结并加该参数。

固定套件准备：

```sh
python -m scripts.r1_reproduction.qualify_r1_full_model prepare \
  --suite-version v2 --cohort-size 16 --time-basis node_monotonic_causal_v2 \
  --tokenizer-json TOKENIZER_JSON --tokenizer-sha256 TOKENIZER_SHA \
  --output FRESH_SUITE_JSON
```

独立语义套件导入：

```sh
python -m scripts.r1_reproduction.r1_independent_semantic import \
  --cohort-size 16 --time-basis node_monotonic_causal_v2 \
  --payload FROZEN_PAYLOAD_JSON --tokenizer-json TOKENIZER_JSON \
  --tokenizer-sha256 TOKENIZER_SHA --output FRESH_INDEPENDENT_SUITE_JSON
```

每个实际 run 使用一份冻结的 binding JSON，字段必须精确如下。占位符替换为已绑定的实际值；节点 IP 为不同的明确 IPv4，hostname 为真实运行 hostname，collector 位于 node0，端口由 root 明确冻结。两节点端口可以相同。`helper_sha256` 填最终 `r1_causal_evidence.py` 文件 SHA，不填整个 binding 的 SHA。

```json
{
  "schema": "r1-causal-clock-binding-v1",
  "run_id": "NEW_DEPLOYMENT_RUN_ID",
  "suite_sha256": "SUITE_FILE_SHA256",
  "deployment_sha256": "DEPLOYMENT_FILE_SHA256",
  "helper_sha256": "FINAL_CAUSAL_HELPER_FILE_SHA256",
  "client_ip": "NODE0_IPV4",
  "nodes": [
    {"node_rank": 0, "ip": "NODE0_IPV4", "port": 18879, "hostname": "ACTUAL_NODE0_HOSTNAME"},
    {"node_rank": 1, "ip": "NODE1_IPV4", "port": 18879, "hostname": "ACTUAL_NODE1_HOSTNAME"}
  ]
}
```

CLI 的 `--binding-sha256` / `--causal-clock-config-sha256` 是保存后 **JSON 文件原始字节 SHA**；ACK 内的 `binding_sha256` 是该对象的 canonical JSON SHA，二者不是同一字段。

## 两节点服务与采集

root 在每个节点的 guard 下分别启动一次只读 ACK 服务。每个服务只接收已知 node0 peer IP、冻结端口和 `/r1-causal/<run_id>` 路径；不接受任意命令或文件路径。`DURATION_SECONDS` 必须覆盖该次采集，属于有限生命期。

```sh
python -m scripts.r1_reproduction.r1_causal_evidence serve \
  --binding BINDING_JSON --binding-sha256 BINDING_FILE_SHA \
  --node-rank NODE_RANK --duration-seconds DURATION_SECONDS
```

`NODE_RANK` 分别为 `0` 和 `1`。单个 run 的所有 cohorts 共用这对服务，服务强制 `begin0 → end0 → begin1 → end1 ...`；不重试已接受的 ACK，也不复用失败/停止 run 的服务状态。short、long、independent，以及 BG/SG 的实际采集各自使用新的 `run_id`、绑定和服务生命周期。

固定套件采集示例：

```sh
python -m scripts.r1_reproduction.qualify_r1_full_model collect \
  --cohort-size 16 --time-basis node_monotonic_causal_v2 \
  --causal-clock-config BINDING_JSON --causal-clock-config-sha256 BINDING_FILE_SHA \
  --suite SUITE_JSON --suite-sha256 SUITE_SHA \
  --deployment DEPLOYMENT_JSON --deployment-sha256 DEPLOYMENT_SHA \
  --backend BACKEND --endpoint MODEL_ENDPOINT --trace-directory TRACE_DIRECTORY \
  --output FRESH_COLLECTION_DIRECTORY --timeout TIMEOUT_SECONDS
```

long 阶段显式加 `--long-correctness`；独立语义阶段改用 `r1_independent_semantic collect`，保留上述 time-basis、cohort、binding 和身份参数，并加既有 `--run-48x2`。使用新的对应 suite/deployment/run；不能通过切换 CLI 把旧采集升级为新证据。

collector 对每个 cohort 顺序执行：**收到双节点 begin ACK → 提交真实 HTTP 请求并收齐全部完整响应 → 收到双节点 end ACK → 开始下一 cohort**。每次 RPC 保留 nonce、cohort/phase、双端实际进程身份和原始时间戳；后台独立时钟探测的时间接近不构成这个调用顺序。

## 验证边界与收尾

同一节点内，原始 native 事件必须满足：

```text
begin.server_send.monotonic_after_ns
    <= event.observed_monotonic_ns
    <= end.server_receive.monotonic_before_ns
```

全部16 rank 的真实 worker PID/start、hostname、boot、time namespace 与偏移、源码身份必须完整且稳定，并与本节点 ACK 服务属于同一时钟域；不同进程的 PID 不要求相同。两个节点之间不比较 namespace 编号或 monotonic 数值，不换算墙钟，也不假设 NTP drift。仍检查精确 job/input/native request/batch ID、ordinal 顺序、完整 token/logits、原数值和语义门禁；缺 ACK、漏 rank、错身份、乱序或超界均失败。

因果 fence 是 **cohort 的外包络**，HTTP 调用自己的起止时间另行保留；不能将 fence 冒充远端精确 HTTP 时间窗或性能时长。

ACK 服务不是 worker drain。当前完成路径应在 API 返回前同步写入 native completion 及其必要 trace；采集与封存必须保留完整原始日志，不能依墙钟或 fence 裁掉迟到事件。如果仍有 native 事件晚于可证明的 end 边界，应显式失败并修正证据收尾路径，不能固定 sleep、删除 trace、自动扩大 end 来通过。累计 `layer_summary` 使用 `last_nonempty_observed_monotonic_ns` 绑定原始实际 forward 的范围；晚写汇总不等于晚做 forward，也不能伪造原观测时间。

旧 `legacy_wall_v1` 路径继续执行原墙钟门禁，旧 blocked/failed 运行保持原状态。1000 ppm 等分析只能标为 conditional 辅助诊断，不能替代本协议、补成正式 pass 或重标旧结果。正式端到端性能的采样与资格另行处理；本协议没有授予正式性能启动或性能结论。

## 关闭原 worker 后封存完整 trace

collector 返回时取得的 trace 只是进度快照，不能单独用于 qualification：此时原 worker 仍可能写出收尾事件。每个 stage 完成后，root 先执行既有 owned cleanup；之后分别在两节点、与原 worker 相同的 PID namespace 内执行 seal。封存前后都只读检查本节点八个已知 worker 的 `(PID, start_ticks)`，不扫描或终止其他进程。

`trace_start` 与 ACK 身份同时记录实际 `pid_namespace={device,inode,link}`。PID namespace 不改变 `same_domain` 的时钟域判定，但它是读取已知 `/proc/<pid>/stat` 来证明 worker 关闭的必要条件。只允许两种闭合状态：已无对应 `/proc`、该 PID 已由不同 start_ticks 的进程复用。原 PID/start_ticks 仍存在时，即使状态为 `Z` 或 `X` 也拒绝封存：主线程退出后，其余线程仍可能共享文件描述符并继续写入。PID 复用只证明原进程已退出，绝不向复用 PID 发信号；身份不明、不可读或 namespace 不符同样拒绝封存。

每节点保存 source-bound `r1-causal-worker-closure-v1` 收据，含实际 observer 身份、全部原 worker 身份及只读结果、观察 monotonic 时间和 helper SHA。离线 finalizer/validator 核验两份完整收据，再对关闭后完整原始 trace 的 SHA、所有 rank/ordinal、因果包络及其余门禁进行验证。迟到事件保留在原始 trace 中，不能借 collector 的早期快照遗漏。这个步骤不依赖宿主 NTP、管理员时钟配置或额外的 sleep；helper/source 与新部署仍须一起重新冻结。

cleanup 完成后，在**每个原节点**分别运行下列 seal，`NODE_RANK` 为 `0` 或 `1`，`ACTUAL_ORIGIN_HOSTNAME` 必须为该原节点真实 hostname，`TRACE_DIRECTORY` 为其完整原始 trace 目录。源文件、suite/deployment 和原始 collector `run.json` 均用本 stage 冻结版本；不要改写原始 run。`--output` 必须使用绝对的新目录。

```sh
python -m scripts.r1_reproduction.finalize_r1_cross_node_trace seal \
  --mode fixed \
  --qualification-tool QUALIFY_R1_FULL_MODEL_PY --qualification-tool-sha256 QUALIFIER_SHA \
  --original-run ORIGINAL_COLLECTOR_RUN_JSON --original-run-sha256 ORIGINAL_RUN_SHA \
  --suite SUITE_JSON --suite-sha256 SUITE_SHA \
  --deployment DEPLOYMENT_JSON --deployment-sha256 DEPLOYMENT_SHA \
  --node-rank NODE_RANK --origin-hostname ACTUAL_ORIGIN_HOSTNAME \
  --trace-directory TRACE_DIRECTORY --output /ABS/FRESH_NODE_SEAL_DIRECTORY
```

执行 seal 后，完整转移两份封存目录：保留各自 `manifest.json` 以及 `raw/` 内**全部**原始 JSONL 和 full-logits 二进制，不能只转移筛选后的事件。原 collector 目录也须保留 `run.json` 及其引用的原 `trace/events.json`。冻结两个 manifest 的文件 SHA，然后在离线汇总节点执行：

```sh
python -m scripts.r1_reproduction.finalize_r1_cross_node_trace finalize \
  --mode fixed \
  --qualification-tool QUALIFY_R1_FULL_MODEL_PY --qualification-tool-sha256 QUALIFIER_SHA \
  --original-run ORIGINAL_COLLECTOR_RUN_JSON --original-run-sha256 ORIGINAL_RUN_SHA \
  --suite SUITE_JSON --suite-sha256 SUITE_SHA \
  --deployment DEPLOYMENT_JSON --deployment-sha256 DEPLOYMENT_SHA \
  --node-manifest NODE0_SEAL_DIRECTORY/manifest.json --node-manifest-sha256 NODE0_MANIFEST_SHA \
  --node-manifest NODE1_SEAL_DIRECTORY/manifest.json --node-manifest-sha256 NODE1_MANIFEST_SHA \
  --output /ABS/FRESH_FINALIZED_DIRECTORY
```

独立语义 stage 的 **seal 与 finalize 都**将 `--mode fixed` 改成 `--mode independent`，并增加以下两项，其他必填参数仍然全部保留：

```sh
--semantic-tool R1_INDEPENDENT_SEMANTIC_PY --semantic-tool-sha256 SEMANTIC_TOOL_SHA
```

semantic 文件旁实际导入的 `qualify_r1_full_model.py` 必须与 `--qualification-tool-sha256` 相符。所有示例仍省略 `--execute`；root 核对计划、身份和新输出目录后再显式执行。finalize 只产生派生完整证据及原始来源收据；后续 BG/SG paired compare、独立语义 score 和实际参数门禁仍须通过，封存成功不能标作整模型或正式性能通过。
