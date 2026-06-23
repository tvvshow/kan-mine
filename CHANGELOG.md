# Changelog

本文件记录面向公开 release / 生产部署的变更。性能数字只引用
`bench/results/` 中可追溯记录；未通过 POSTCHECK、controlled benchmark 和
pool accepted 的实验不标记为 production recommended。

---

## v1.2.18 — production multi-GPU portable release candidate

状态：待 Linux/CUDA release workflow 构建与 GPU L2/L3 验证后发布。

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
- 文档按实际 fatbin/kernel 修正硬件范围：当前 production release 从 `sm_75`
  开始覆盖；V100/V100S / Volta / `sm_70` 不属于当前支持范围。
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
