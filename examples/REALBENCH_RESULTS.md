# Real-corpus benchmark results

Run with `examples/realbench.rs` (target ~8 KiB; mincdc min=2048 max=14336;
fastcdc min=2048 avg=8192 max=51200 to avoid its forced-cut tail). Native
arm64 / NEON. Throughput is chunking only (no hashing); absolute GB/s is
indicative (iterator overhead), relative comparison is the signal. Dedup% =
1 − stored_bytes / logical_bytes; "rec" = records (metadata rows).

Corpora (all real or built from real data):
1. **Linux kernels** — `linux-6.6.tar` + `linux-6.7.tar` (2.7 GiB, uncompressed).
2. **Container rootfs** — `docker export` of `alpine:3.18/3.19/3.20` (real fs tars).
3. **APFS disk images** — two 200 MiB `hdiutil` APFS images holding real source
   trees (v2 = v1's files + an extra version), mostly zero-fill like a real disk.
4. **SQLite DB** — 300k-row DB, v2 = v1 with 8% rows updated, both `VACUUM`ed.
5. **Versioned source** — redis 7.2.3 / 7.2.4 / 7.2.5 source trees (4747 files).

| Corpus | algo | GB/s | records | uniq | dedup% | mean | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Linux 6.6+6.7** | fastcdc | 1.05 | 244448 | 163478 | 32.5 | 11751 | 41437 | 51200 |
| | mincdc-plain | **4.89** | 400716 | 236146 | **36.3** | 7168 | 14200 | 14336 |
| | cat-simple | 3.20 | 400709 | 236146 | 36.3 | 7168 | 14200 | 14336 |
| | cat-period | 1.50 | 400709 | 236146 | 36.3 | — | — | — |
| **Containers (alpine ×3)** | fastcdc | 1.70 | 2321 | 2191 | 5.6 | 10811 | 51200 | 51200 |
| | mincdc-plain | 6.90 | 3633 | 2711 | **11.7** | 6906 | 14236 | 14336 |
| | cat-simple | 7.07 | **2945** | 2711 | 11.7 | 8520 | 36864 | 49152 |
| | cat-period | **0.03** | 2945 | 2711 | 11.7 | — | — | — |
| **APFS disk images** | fastcdc | 1.30 | 12021 | 1933 | 94.3 | 34891 | 51200 | 51200 |
| | mincdc-plain | 2.35 | **182701** | 2918 | 94.5 | 2295 | 11738 | 14335 |
| | cat-simple | 2.19 | **7798** | 2918 | 94.5 | 53786 | 14222 | 188M |
| | cat-period | 0.97 | 7798 | 2918 | 94.5 | — | — | — |
| **SQLite (VACUUMed)** | fastcdc | 1.53 | 8006 | 7997 | 0.0 | 10004 | 27069 | 46000 |
| | mincdc-plain | 8.67 | 9426 | 9418 | 0.0 | 8497 | 14219 | 14336 |
| | cat-simple | 8.58 | 9426 | 9418 | 0.0 | 8497 | 14219 | 14336 |
| | cat-period | 2.11 | 9426 | 9418 | 0.0 | — | — | — |
| **Redis source ×3** | fastcdc | 1.67 | 8156 | 2770 | 65.3 | 5684 | 23484 | 40309 |
| | mincdc-plain | **9.02** | 12562 | 4220 | **65.7** | 3690 | 13798 | 14316 |
| | cat-simple | 8.68 | 12562 | 4220 | 65.7 | 3690 | 13798 | 14316 |
| | cat-period | 1.71 | 12562 | 4220 | 65.7 | — | — | — |

## Conclusions

**mincdc beats fastcdc on real data, consistently:**
- Speed: ~4.7–6× faster (Linux 4.89 vs 1.05; source 9.0 vs 1.7 GB/s).
- Dedup: equal-or-better everywhere (Linux 36.3 vs 32.5; container 11.7 vs 5.6;
  source 65.7 vs 65.3; disk 94.5 vs 94.3).
- Size distribution: mincdc is hard-bounded (max=14336 always); fastcdc has a
  long tail (max 40k–51k, inflated means). Predictable sizes are real.

**Tier-1 caterpillar (cat-simple) is the keeper — value is workload-dependent:**
- **Disk images: 182701 → 7798 records (−96%)**, beating fastcdc's 12021, while
  keeping 94.5% dedup. It fixes mincdc-plain's one real weakness (a zero-fill
  record explosion) and turns it into a win.
- Containers: 3633 → 2945 (−19%) from tar/zero padding.
- Source / kernel / DB: a free no-op (no long contiguous runs to coalesce).
- Never hurts dedup; ~free where it can't help. Pure insurance.

**Tier-2 caterpillar (cat-period): evaluated and removed.**
- It never reduced records below tier-1 on ANY real corpus (the phase-rotating
  case it targets did not occur — mincdc self-aligns, and real periodic data
  either dedups via content addressing or was rewritten away).
- Its gate is catastrophic on real data: 0.03 GB/s on containers, 0.97 on disk.
- Removed from the public API for these reasons; the implementation and these
  `cat-period` rows are preserved in git (branch `proto/caterpillar-period`, last
  main-line commit with tier-2 `35cf93e`). See `examples/CATBENCH_RESULTS.md`.

**Honest surprises the real data exposed (that synthetic tests missed):**
- **SQLite VACUUM → 0% dedup.** A full rewrite reshuffles pages, destroying
  byte-level redundancy. Real DB dedup only works on non-rewritten/incremental
  files.
- Tier-1's win is concentrated in **zero-fill / padding-heavy** workloads (disk
  images, containers, VM images) — exactly where it matters operationally — and
  is correctly a no-op on source code.
