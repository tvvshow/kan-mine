# Changelog

本文件记录面向公开 release / 生产部署的变更。性能数字只引用
`bench/results/` 中可追溯记录；未通过 POSTCHECK、controlled benchmark 和
pool accepted 的实验不标记为 production recommended。

---

## v1.2.21 — multi-pool failover + monitoring API

状态：待 Linux/CUDA release workflow 构建。功能已在 2× RTX 2080 Ti 实测验证。

### Pool / 可靠性

- **多池故障转移**：`--pool` 现在可重复指定（主池优先，其余为备池）。每个 GPU lane
  按顺序尝试端点，连接并授权成功的第一个就开挖；全部失败则交由 supervisor 重启整条
  lane，再从主池重新扫起（主池恢复后自动回切）。挖矿/提交主体代码未改动，单池行为不变。
  实测：死主池 `127.0.0.1:9999` + ssl 备池 → connect failed → endpoint 2/2 →
  authorize ok → 开挖。

### 监控

- **HTTP/JSON 监控 API**：`--api-port N` 暴露每卡与聚合的算力、accepted/rejected，
  以及 NVML 温度/转速/功耗，便于 HiveOS / mmpOS / curl 等矿场面板对接。多进程感知：
  每个 lane 把自身 stats 写成 `KAN_API_DIR/gpuN.json`（execvp 会清空 mmap，故用文件
  IPC 经环境变量传递），多卡由 parent、单卡由本进程跑一个轻量 HTTP server 聚合。
  实测：`GET /` 返回 `{total, gpus[2]}`，total 34.98 = 19.43 + 15.55 TH/s，含实时
  NVML 读数。

---

## v1.2.20 — Turing / sm_75 support via WMMA flavor

状态：待 Linux/CUDA release workflow 构建与 Turing 实机 pool 验证后发布。

### GPU support

- **新增 Turing / `sm_75`（RTX 20 系）支持**：通过专用便携包
  `kan-portable-linux-x64-sm75.tar.gz`。generic（CUTLASS）包仍无法在 Turing 启动
  —— Sm80 内核需要 cp.async（sm_80+）和 ~89KB dynamic shared memory，而 Turing
  每 block 上限 64KB 且没有 cp.async。sm75 包改用 WMMA 内核 `tc_block.cu`
  （int8 16x16x16，32KB **静态** shared memory，`__pipeline_memcpy_async` 在
  sm_75 上自动降级为同步拷贝），可在 RTX 20 / Titan RTX / Quadro RTX / T4 运行。
- `build.sh` 新增 `KERNEL` 开关（auto/wmma/cutlass）：`ARCH=sm_75` 自动选 WMMA；
  `KERNEL=wmma` 即使存在 CUTLASS 头也强制 tc_block。构建写出 `build/BUILD_KERNEL`，
  打包记入 `BUILD_INFO.txt` 的 `kernel:` 字段。
- `install_kan.sh` 选包：`sm_75 -> kan-portable-linux-x64-sm75.tar.gz`，新增
  `--force-sm75`。`.cnb.yml` release matrix 增加 `package-sm75-turing` 构建阶段与
  附件、SHA256SUMS。
- **GPU-resident draw pipeline 接入 WMMA 路径**：`gpu_prep.cu`（GPU 侧 RNG + BLAKE3
  tree + noise，纯 CUDA、无 cp.async/Tensor Core，Turing 可跑）原先只链入 CUTLASS
  路径，现在 WMMA 路径也编链它；`tc_block.cu` 暴露 `tc_alloc_bufs` 并在 `a_noised=NULL`
  时跳过 H2D。每 draw 的 prep 从 ~2730ms（CPU）降到 ~18ms（GPU），消除多卡共享主机
  CPU 时的 prep 争抢。
- 状态：**Candidate**。RTX 2080 Ti 实测 `ARCH=sm_75 KERNEL=wmma` 构建 +
  real-cfg POSTCHECK ok=1、emitted proof（与 Ampere 同尺寸）。内核约 20 TH/s；
  GPU-resident pipeline 使单卡 wall 16.1→19.4 TH/s、**双卡聚合 26.1→37.2 TH/s
  （缩放 1.65×→1.92×，近线性，util 100%）**，明显低于 Ampere+ 属预期。live pool
  accepted / 正式 `bench/results` 记录待补。
- GTX 10 系 / Pascal（`sm_60`/`sm_61`）无 int8 Tensor Core，需要 DP4A 路径，列入
  后续优化，本版本不含。

---

## v1.2.19 — operator polish / checksum / service candidate

状态：待 Linux/CUDA release workflow 构建与 GPU L2 验证后发布。

### Packaging / operator experience

- Release matrix 增加 `SHA256SUMS` 附件；`package_portable.sh` 生成 tarball
  校验和，CNB tag pipeline 在上传前统一生成 generic、sm86-g8 与
  `install_kan.sh` 的 checksum。
- `install_kan.sh` 增加 CLI 参数：`--version`、`--dest`、`--base-url`、
  `--force-generic`、`--force-sm86-g8`、`--force-package`、`--gpu-sm`、
  `--dry-run`、`--no-status`；同时增加 glibc 版本提示、SHA256SUMS 校验和
  安装后 `status.sh` 快照。
- 便携包新增可选 systemd/logrotate 生产部署模板：`install_service.sh`、
  `kan.service`、`kan.logrotate`。默认生成 `kan.env`，由 operator 填写
  `KAN_WALLET` 后启用服务。

### GPU support wording

- 将 `sm_75` / Turing 从“production compatibility”改为
  **unsupported in current production package**：RTX 2080 Ti 实测当前内核因
  shared-memory 要求 kernel launch 失败；generic fatbin 不再包含 `sm_75`。

---

## v1.2.18 — production multi-GPU portable release

状态：已发布；main、gpu-verify 与 tag release build 均通过，generic / sm86-g8 / install 脚本资产已上传。

### Production runtime

- **Async share submit**：pool 模式下 fresh share proof 生成后先入队，由
  submit worker 发送并等待矿池响应；mining 主循环立即进入下一轮，避免
  `submit_wait` 网络延迟阻塞 GPU mining。
- **Stale proof early-abort**：如果新 pool job 在 win/proof 期间到达，过期
  proof 会在 CPU rederive、POSTCHECK 或 Merkle 阶段之间提前退出，避免为不可
  提交的 stale share 继续消耗 CPU。
- 保持 correctness gate 不变：fresh share 仍必须通过 CPU POSTCHECK 后才会
  进入提交队列。
- pool mode now requires an explicit `--wallet ADDR[.WORKER]`; public packages
  no longer carry a real default wallet address.
- **Single-machine multi-GPU**：pool 模式将单机单卡和单机多卡列为正式生产
  运行能力。未设置 `CUDA_VISIBLE_DEVICES` 时自动使用所有检测到的 GPU
  （parent supervisor + 每 GPU 一个隔离 lane 进程，各自独立 stratum 连接，
  以同一 worker 名聚合）。新增 `--devices 0,1,3` 选择物理 GPU 子集；外部
  `CUDA_VISIBLE_DEVICES` 时 miner 尊重它并禁用 auto fanout。`--devices` 与
  `CUDA_VISIBLE_DEVICES` 互斥；`--devices` 仅 pool 模式有效。任一 lane 异常
  退出时 supervisor 停止其余 lane 整体退出，交给 `run.sh` 自动重启。
  诚实声明：统一父进程 stats / 单一共享 stratum session 仍是未来项，未完成。

### Packaging / operator docs

- portable package `README.txt` / `RELEASE_NOTES.txt` 增加 runtime behavior
  说明：pool auto-restart、async submit、stale proof abort、multi-GPU auto fanout。
- portable `run.sh` 启动 banner 增强：检测到多卡时明确提示 multi-GPU auto
  fanout 默认启用，并说明 `--devices` / `CUDA_VISIBLE_DEVICES` 限制方式；
  不改变现有 `KAN_RESTART` pool auto-restart 行为。
- `status.sh` 增加 runtime event 摘要：
  - `async_submit_worker_seen`
  - `stale_proof_aborts`
- package 内复制 `CHANGELOG.md`，便于下载包离线审计。
- 新增 CNB on-demand GPU 验证入口：
  - `.cnb.yml` 增加 `web_trigger_gpu_verify`；
  - `.cnb/web_trigger.yml` 增加 main 分支 **GPU 验证** 按钮；
  - `ci_gpu_verify.sh` 执行 build、`run_test.sh`、real-cfg POSTCHECK、
    generic portable package build 和 package POSTCHECK。
  - 默认不连接矿池；只有显式设置 `GPU_VERIFY_POOL_SECONDS>0` 且提供
    `KAN_WALLET` 时才做 live pool smoke。

### GPU profiles / public performance wording

- README 和 `GPU_PROFILES.md` 明确区分：
  - generic compatibility package；
  - `sm86-g8` production tuned package；
  - RTX 5090 / `sm_120` CUDA 12 generic PTX fallback baseline。
- 文档按实际 fatbin/kernel 修正硬件范围：Volta / `sm_70` 与 Turing / `sm_75`
  不属于当前支持范围；正式 production 推荐从已验证的 `sm_86` tuned package 起。
- RTX 5090 / `sm_120` 不自动选择 tuned 包；CUDA 13 native `sm_120` package
  只有在完成 POSTCHECK、controlled benchmark、live pool accepted 和稳定性验证后
  才能升级为 production recommended。

### Validation required before tagging

必须通过：

```text
L0: bash check_release_profiles.sh
L1: WITH_AB=0 bash package_portable.sh
L1: WITH_AB=0 ARCH=sm_86 GROUPM=8 KSTAGES=3 PACKAGE_FLAVOR=sm86-g8 bash package_portable.sh
L2: GPU smoke POSTCHECK ok=1
L3: pool accepted > 0, rejected not abnormal, submit_timeout no storm
```

推荐 tag message：

```text
v1.2.18 — production multi-GPU portable release

- pool: async share submit worker prevents submit_wait from blocking mining
- proof: abort stale proof assembly when a newer pool job arrives
- runtime: single-machine single-GPU and multi-GPU auto fanout with --devices scoping
- package: include CHANGELOG, runtime behavior notes, status runtime events
```

---

## v1.2.17 — production validation baseline

- Switched production smoke tests to real Pearl config so CI validates the release
  path against the same dimensions used on mainnet.
- Prepared the v1.2.17 release pipeline and GPU validation gates that v1.2.18
  builds on.

---

## v1.2.16 — CI locale fix + honest performance docs

- Release-profile static check no longer depends on locale-specific sort order.
- README performance wording no longer presents RTX 4090/5090 experimental
  numbers as shipped release guarantees.
- Default release assets remain:
  - `kan-portable-linux-x64.tar.gz`
  - `kan-portable-linux-x64-sm86-g8.tar.gz`
  - `install_kan.sh`

---

## v1.2.15 — generic portable baseline used for RTX 5090 validation

- Public generic portable package validated on RTX 5090 / `sm_120` via CUDA 12
  `compute_90` PTX fallback.
- Recorded benchmark references:
  - `bench/results/2026-06-22_rtx5090_sm120_vps.md`
  - `bench/results/2026-06-23_rtx5090_sm120_vps106.md`
- This is a healthy compatibility baseline, not a native Blackwell tuned profile.
