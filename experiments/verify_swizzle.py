# verify_swizzle.py — one-shot CPU proof of the SWIZZLE=1 smem layout in
# tc_deep_pipeline.cu (2026-06-10). Checks, for RSUB in {32,64,128}:
#   1. SELF-CONSISTENCY: cp.async writes and ldmatrix reads both address through
#      SWZ -> every 16B chunk is read back from where it was written (bijection
#      within each row, identity on data).
#   2. BANK CONFLICTS: per ldmatrix 8-row phase (A .x4 rows 16-aligned, B .x2
#      rows 8-aligned) count max ways per 4-bank group, linear vs swizzled.
#      Expect linear: RSUB=32->2, 64->4, 128->8 ; swizzled: all 1.
#   3. WRITE SIDE: per 8-consecutive-thread cp.async group, banks must stay
#      uniform (max ways 1) with swizzle on.
# Run: python experiments/verify_swizzle.py   (pure stdlib, no GPU needed)

def swz(r, o, rsub):
    # mirrors: #define SWZ(r,o) ((o) ^ (((((r)*RSUB) >> 7) & (RSUB/16-1)) << 4))
    return o ^ ((((r * rsub) >> 7) & (rsub // 16 - 1)) << 4)

def bank_group(addr):
    # 32 banks x 4B = 128B wavefront; a 16B chunk occupies one of 8 4-bank groups
    return (addr // 16) % 8

fail = 0
for rsub in (32, 64, 128):
    nc = rsub // 16
    # --- 1. self-consistency: SWZ is a bijection on columns within every row ---
    for r in range(256):
        cols = [swz(r, c * 16, rsub) // 16 for c in range(nc)]
        assert sorted(cols) == list(range(nc)), f"RSUB={rsub} r={r}: not a permutation {cols}"
        for c in range(nc):  # read(SWZ) of (r,c) hits exactly the chunk write(SWZ) put there
            assert swz(r, c * 16, rsub) == swz(r, c * 16, rsub)  # same macro both sides
    # --- 2. ldmatrix read phases: 8 consecutive rows, fixed 16B column ---
    worst_lin, worst_swz = 0, 0
    for r0 in range(0, 256, 8):           # A .x4 phases are 16-aligned, B .x2 8-aligned; 8 covers both
        for c in range(nc):
            lin = [bank_group(r * rsub + c * 16) for r in range(r0, r0 + 8)]
            sw  = [bank_group(r * rsub + swz(r, c * 16, rsub)) for r in range(r0, r0 + 8)]
            worst_lin = max(worst_lin, max(lin.count(g) for g in set(lin)))
            worst_swz = max(worst_swz, max(sw.count(g) for g in set(sw)))
    # --- 3. cp.async write side: thread e -> row e//nc, col e%nc; phase = 8 threads ---
    worst_w = 0
    for e0 in range(0, 256, 8):
        sw = [bank_group((e // nc) * rsub + swz(e // nc, (e % nc) * 16, rsub)) for e in range(e0, e0 + 8)]
        worst_w = max(worst_w, max(sw.count(g) for g in set(sw)))
    ok = worst_swz == 1 and worst_w == 1
    fail += 0 if ok else 1
    print(f"RSUB={rsub:3d}: ldmatrix conflict ways linear={worst_lin} swizzled={worst_swz} | "
          f"cp.async write ways swizzled={worst_w} | {'PASS' if ok else 'FAIL'}")

print("ALL PASS" if fail == 0 else f"{fail} FAILURES")
raise SystemExit(fail)
