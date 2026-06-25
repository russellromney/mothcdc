# mincdc vs mincdc+caterpillar — speed / latency / metadata

`examples/catbench.rs`, best-of-5, native arm64/NEON, quiet system. min=2048
max=14336. meta = records × 48 B (32 B fingerprint + offset/len). No fastcdc.

| Corpus (size) | algo | GB/s | Δspeed | ms | ns/chunk | records | Δrec | meta | meta% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **redis source** (44 MiB) | plain | 8.86 | — | 5.2 | 417 | 12562 | — | 0.6 MiB | 1.30 |
| | cat-simple | 8.79 | −1% | 5.3 | 420 | 12562 | +0% | 0.6 MiB | 1.30 |
| | cat-period | 1.71 | −81% | 27.1 | 2154 | 12562 | +0% | 0.6 MiB | 1.30 |
| **sqlite** (76 MiB) | plain | 8.84 | — | 9.1 | 962 | 9426 | — | 0.4 MiB | 0.56 |
| | cat-simple | 8.82 | −0% | 9.1 | 963 | 9426 | +0% | 0.4 MiB | 0.56 |
| | cat-period | 2.12 | −76% | 37.8 | 4015 | 9426 | +0% | 0.4 MiB | 0.56 |
| **containers** (24 MiB) | plain | 6.00 | — | 4.2 | 1151 | 3633 | — | 0.2 MiB | 0.69 |
| | cat-simple | 7.09 | **+18%** | 3.5 | 1203 | 2945 | **−19%** | 0.1 MiB | 0.56 |
| | cat-period | 0.03 | −99% | 836 | 283729 | 2945 | −19% | 0.1 MiB | 0.56 |
| **disk images** (400 MiB) | plain | 2.37 | — | 177 | 970 | 182701 | — | 8.4 MiB | 2.09 |
| | cat-simple | 2.20 | −7% | 191 | 24433 | **7798** | **−96%** | 0.4 MiB | 0.09 |
| | cat-period | 0.98 | −59% | 427 | 54806 | 7798 | −96% | 0.4 MiB | 0.09 |
| **linux 6.6+6.7** (2.7 GiB) | plain | 7.41 | — | 388 | 968 | 400716 | — | 18.3 MiB | 0.67 |
| | cat-simple | 7.38 | −0% | 389 | 972 | 400709 | −0% | 18.3 MiB | 0.67 |
| | cat-period | 1.79 | −76% | 1606 | 4009 | 400709 | −0% | 18.3 MiB | 0.67 |

## mincdc vs mincdc+cat-simple (tier 1) — the real comparison

**Speed/latency: free, sometimes faster, worst case −7%.**
- No-coalesce data (source, sqlite, kernel): −0% to −1% — within noise. Truly free.
- Heavy-coalesce data (containers): **+18%** — fewer records to emit.
- Disk images: −7% — it does real compare work over giant zero runs, but for a
  −96% metadata payoff.
(The earlier single-pass Linux −35% was measurement noise on a busy box; clean
best-of-5 shows −0%.)

**Metadata: 0 where it can't help, up to −96% where it can.**
- Source / sqlite / kernel: identical (no contiguous identical runs).
- Containers: −19% (tar/zero padding).
- Disk images: **−96%** (182701 → 7798 records; 8.4 MiB → 0.4 MiB; 2.09% → 0.09%).
- Dedup (unique records) is unchanged in every case — the win is metadata, not
  stored bytes.

**Verdict:** tier-1 is speed-neutral-to-positive and never increases metadata.
It's a free pass that erases mincdc's zero-fill record explosion on disk/VM/
container data.

## cat-period (tier 2) — evaluated and removed

−76% to −99% throughput on every corpus (283 µs/chunk on containers!) and never
fewer records than tier-1. The phase-rotating case it targets did not occur in
any real corpus: mincdc self-aligns its boundaries to most periods, so tier-1
already collapses periodic data whenever a period multiple fits in `[min, max]`.

Because it never earned its cost, the second tier was removed from the public API
(no more `CaterpillarChunker::with_period_detection`). The full implementation —
content-defined period detection, Booth-rotation canonicalization, the
`Segment::Periodic` variant, and these benchmarks — is preserved in git history:

- branch `proto/caterpillar-period`
- last commit that still carried tier-2 on the main line: `35cf93e`

The `cat-period` rows above are kept as the record of why tier-1 alone is the
shipped design. (The earlier rows were produced with tier-2 still present.)
