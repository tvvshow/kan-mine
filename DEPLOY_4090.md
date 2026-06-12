# 4090 优化部署指南

## 快速开始（3 分钟）

```bash
# 1. 上传优化包到 4090 box
scp D:\mybitcoin\3\peral\optimization_kit.tar.gz ubuntu@117.50.47.40:/home/ubuntu/

# 2. SSH 到 4090 box
ssh ubuntu@117.50.47.40

# 3. 解压并测试
cd kan
tar xzf ../optimization_kit.tar.gz
chmod +x bench/*.sh
bash bench/quick_test.sh
```

## 预期结果

### 基线（当前）
- GROUPM=8, TB 128×256
- 260 TH/s kernel, 270ms/draw

### 目标
- **GROUPM=16**：如果 <260ms/draw → 280+ TH/s (+7%)
- **SMALL_TILE=1**：如果 ~200ms/draw → 350+ TH/s (+35%)
- **组合**：如果 ~180ms/draw → 400+ TH/s (+54%)

## 手动测试单个配置

```bash
# 测试 GROUPM=16
cd kan
GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"

# 测试 SMALL_TILE (2 TB/SM occupancy)
SMALL_TILE=1 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"

# 测试组合
SMALL_TILE=1 GROUPM=16 ./build.sh
./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | grep "ms/draw"
```

## 部署到生产

找到最快配置后（例如 SMALL_TILE=1 GROUPM=16）：

```bash
# 1. 用最优参数重新编译
cd kan
SMALL_TILE=1 GROUPM=16 ./build.sh

# 2. 停止当前挖矿
pkill -9 kan

# 3. 启动新版本
nohup ./build/kan --algo pearl \
  --pool stratum+tcp://prl.kryptex.network:7048 \
  --wallet prl1patz2m...apmv.pm \
  > kan.log 2>&1 &

# 4. 监控日志
tail -f kan.log
```

## 技术细节

详见 `OPTIMIZATION_PLAN.md`

## 文件说明

- `build.sh` — 支持 GROUPM 和 SMALL_TILE 环境变量
- `src/tc_cutlass_v2.cu` — 内核源码，已添加 SMALL_TILE 支持
- `bench/quick_test.sh` — 快速测试脚本（3个配置）
- `bench/sweep_4090.sh` — 完整扫描脚本（所有配置）
- `OPTIMIZATION_PLAN.md` — 详细优化分析

## 回滚

如果新配置不稳定：

```bash
cd kan
unset GROUPM SMALL_TILE
./build.sh   # 恢复默认 GROUPM=8, TB 128×256
# 重启 kan
```
