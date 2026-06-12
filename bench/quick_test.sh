#!/usr/bin/env bash
# 快速单项测试 — 在 4090 box 上运行
set -e

cd "$(dirname "$0")/.."

echo "=== Quick optimization test on 4090 ==="
echo "Baseline: 260 TH/s (270ms/draw, GROUPM=8, TB 128×256)"
echo ""

# 只测试最有希望的两个配置
tests=(
  "GROUPM=16"
  "SMALL_TILE=1"
  "SMALL_TILE=1:GROUPM=16"
)

for test in "${tests[@]}"; do
  IFS=':' read -ra PARAMS <<< "$test"
  echo "--- Testing: $test ---"

  # 设置环境变量
  for param in "${PARAMS[@]}"; do
    export $param
  done

  # 构建
  ./build.sh 2>&1 | grep -E "CUTLASS at|BUILD OK" || { echo "BUILD FAILED"; continue; }

  # 运行3次draw测速
  echo -n "Running: "
  timeout 90 ./build/plainproof_gen --cfg real --mine --batch 3 2>&1 | \
    tee /tmp/test.log | grep -oP '\d+ draws.*?(\d+\.\d+) ms/draw' | tail -1 || echo "TIMEOUT"

  # 提取关键数据
  if grep -q "POSTCHECK ok=1" /tmp/test.log; then
    echo "  ✓ POSTCHECK passed"
  else
    echo "  ✗ POSTCHECK failed or not found"
  fi

  # 清理环境变量
  unset GROUPM SMALL_TILE
  echo ""
done

echo "Restoring baseline..."
./build.sh >/dev/null 2>&1
echo "Done. If any config shows <250ms/draw, deploy it to production!"
