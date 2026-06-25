use std::arch::x86_64::*;
use std::sync::LazyLock;

use crate::scalar;

const PREFETCH_DIST: usize = 16384;

// The avx512_impl, avx2_impl or sse41_impl finds a position which has the
// global minimizer within the next four bytes.

/// # Safety
/// AVX-512F must be available.
#[target_feature(enable = "avx512f")]
unsafe fn avx512_impl<const SHOULD_HASH: bool>(
    bytes: &[u8],
    multiplier: u32,
    addend: u32,
) -> usize {
    if bytes.len() < 64 + 3 {
        return unsafe { avx2_impl::<SHOULD_HASH>(bytes, multiplier, addend) };
    }

    let vmul = _mm512_set1_epi32(multiplier as i32);
    let vadd = _mm512_set1_epi32(addend as i32);
    let mut accum_lo = _mm512_set1_epi64(u64::MAX as i64);
    let mut accum_hi = _mm512_set1_epi64(u64::MAX as i64);

    let mut offset = 0;
    unsafe {
        while offset + 4 <= bytes.len() {
            if offset + 64 + 3 > bytes.len() {
                offset = bytes.len() - 64 - 3;
            }

            // wrapping_add: offset + PREFETCH_DIST routinely lands past the end
            // of the search slice. The address is only ever fed to _mm_prefetch
            // (which faults silently), but `ptr::add` past the allocation is UB
            // in Rust even without a deref, so use wrapping_add to stay sound.
            _mm_prefetch::<_MM_HINT_T0>(bytes.as_ptr().wrapping_add(offset + PREFETCH_DIST).cast());

            let mut v0 = _mm512_loadu_si512(bytes.as_ptr().add(offset).cast());
            let mut v1 = _mm512_loadu_si512(bytes.as_ptr().add(offset + 1).cast());
            let mut v2 = _mm512_loadu_si512(bytes.as_ptr().add(offset + 2).cast());
            let mut v3 = _mm512_loadu_si512(bytes.as_ptr().add(offset + 3).cast());

            if SHOULD_HASH {
                v0 = _mm512_mullo_epi32(v0, vmul);
                v1 = _mm512_mullo_epi32(v1, vmul);
                v2 = _mm512_mullo_epi32(v2, vmul);
                v3 = _mm512_mullo_epi32(v3, vmul);
                v0 = _mm512_add_epi32(v0, vadd);
                v1 = _mm512_add_epi32(v1, vadd);
                v2 = _mm512_add_epi32(v2, vadd);
                v3 = _mm512_add_epi32(v3, vadd);
            }

            let m01 = _mm512_min_epu32(v0, v1);
            let m23 = _mm512_min_epu32(v2, v3);
            let m0123 = _mm512_min_epu32(m01, m23);

            // We use the 64-bit minimum to compute the (val, offset) argmin
            // by interleaving values and offsets.
            let voffset = _mm512_set1_epi32(offset as i32);
            let pairs_lo = _mm512_unpacklo_epi32(voffset, m0123);
            let pairs_hi = _mm512_unpackhi_epi32(voffset, m0123);
            accum_lo = _mm512_min_epu64(accum_lo, pairs_lo);
            accum_hi = _mm512_min_epu64(accum_hi, pairs_hi);
            offset += 64;
        }

        // Add the corresponding relative offsets for each 64-bit subregister.
        static SUBREG_OFFSET_LO: [u64; 8] = [0, 4, 16, 20, 32, 36, 48, 52];
        static SUBREG_OFFSET_HI: [u64; 8] = [8, 12, 24, 28, 40, 44, 56, 60];
        accum_lo = _mm512_add_epi64(
            accum_lo,
            _mm512_loadu_epi64(SUBREG_OFFSET_LO.as_ptr().cast()),
        );
        accum_hi = _mm512_add_epi64(
            accum_hi,
            _mm512_loadu_epi64(SUBREG_OFFSET_HI.as_ptr().cast()),
        );
        accum_lo = _mm512_min_epu64(accum_lo, accum_hi);

        let mut out = [0u64; 8];
        _mm512_storeu_si512(out.as_mut_ptr().cast(), accum_lo);
        out.iter().copied().min().unwrap() as u32 as usize
    }
}

/// # Safety
/// AVX2 must be available.
#[target_feature(enable = "avx2")]
unsafe fn avx2_impl<const SHOULD_HASH: bool>(bytes: &[u8], multiplier: u32, addend: u32) -> usize {
    if bytes.len() < 32 + 3 {
        return sse41_impl::<SHOULD_HASH>(bytes, multiplier, addend);
    }

    // AVX2 has no unsigned comparison so add i32::MIN, shifting 0 to i32::MIN.
    let vmul = _mm256_set1_epi32(multiplier as i32);
    let vadd = _mm256_set1_epi32((addend as i32).wrapping_add(i32::MIN));

    let mut offset = 0;
    let mut min_val = _mm256_set1_epi32(i32::MAX);
    let mut min_offset = _mm256_set1_epi32(0);

    let mut body = |offset: &mut usize| unsafe {
        let mut v0 = _mm256_loadu_si256(bytes.as_ptr().add(*offset).cast());
        let mut v1 = _mm256_loadu_si256(bytes.as_ptr().add(*offset + 1).cast());
        let mut v2 = _mm256_loadu_si256(bytes.as_ptr().add(*offset + 2).cast());
        let mut v3 = _mm256_loadu_si256(bytes.as_ptr().add(*offset + 3).cast());

        let m0123s;
        if SHOULD_HASH {
            // We fused the unsigned -> signed conversion into the adds here.
            v0 = _mm256_mullo_epi32(v0, vmul);
            v1 = _mm256_mullo_epi32(v1, vmul);
            v2 = _mm256_mullo_epi32(v2, vmul);
            v3 = _mm256_mullo_epi32(v3, vmul);
            v0 = _mm256_add_epi32(v0, vadd);
            v1 = _mm256_add_epi32(v1, vadd);
            v2 = _mm256_add_epi32(v2, vadd);
            v3 = _mm256_add_epi32(v3, vadd);
            let m01s = _mm256_min_epi32(v0, v1);
            let m23s = _mm256_min_epi32(v2, v3);
            m0123s = _mm256_min_epi32(m01s, m23s);
        } else {
            // We do a single conversion here after computing the unsigned minimum.
            let m01u = _mm256_min_epu32(v0, v1);
            let m23u = _mm256_min_epu32(v2, v3);
            let m0123u = _mm256_min_epu32(m01u, m23u);
            m0123s = _mm256_add_epi32(m0123u, _mm256_set1_epi32(i32::MIN));
        }

        let voffset = _mm256_set1_epi32(*offset as i32);
        let better = _mm256_cmpgt_epi32(min_val, m0123s);
        min_val = _mm256_min_epi32(min_val, m0123s);
        min_offset = _mm256_blendv_epi8(min_offset, voffset, better);
        *offset += 32;
    };

    unsafe {
        while offset + 64 + 4 <= bytes.len() {
            // See note in avx512_impl: wrapping_add keeps the OOB prefetch
            // address computation sound.
            _mm_prefetch::<_MM_HINT_T0>(bytes.as_ptr().wrapping_add(offset + PREFETCH_DIST).cast());

            // Manually unrolled twice.
            body(&mut offset);
            body(&mut offset);
        }
        while offset + 4 <= bytes.len() {
            if offset + 32 + 3 > bytes.len() {
                offset = bytes.len() - 32 - 3;
            }

            body(&mut offset);
        }

        // Add the corresponding relative offsets for each 32-bit subregister
        // and interleave to 64-bit (value, offset) pairs.
        static SUBREG_OFFSET: [u32; 8] = [0, 4, 8, 12, 16, 20, 24, 28];
        min_offset = _mm256_add_epi32(
            min_offset,
            _mm256_loadu_si256(SUBREG_OFFSET.as_ptr().cast()),
        );
        let pairs_lo = _mm256_unpacklo_epi32(min_offset, min_val);
        let pairs_hi = _mm256_unpackhi_epi32(min_offset, min_val);

        let mut out = [0i64; 8];
        _mm256_storeu_si256(out.as_mut_ptr().cast(), pairs_lo);
        _mm256_storeu_si256(out.as_mut_ptr().add(4).cast(), pairs_hi);
        out.iter().copied().min().unwrap() as u32 as usize
    }
}

/// # Safety
/// SSE4.1 must be available.
#[target_feature(enable = "sse4.1")]
fn sse41_impl<const SHOULD_HASH: bool>(bytes: &[u8], multiplier: u32, addend: u32) -> usize {
    if bytes.len() < 16 + 3 {
        return scalar::argmin_u32_overlapping_hashed::<SHOULD_HASH>(bytes, multiplier, addend);
    }

    // SSE2 has no unsigned comparison so add i32::MIN, shifting 0 to i32::MIN.
    let vmul = _mm_set1_epi32(multiplier as i32);
    let vadd = _mm_set1_epi32((addend as i32).wrapping_add(i32::MIN));

    let mut offset = 0;
    let mut min_val = _mm_set1_epi32(i32::MAX);
    let mut min_offset = _mm_set1_epi32(0);

    let mut body = |offset: &mut usize| unsafe {
        let mut v0 = _mm_loadu_si128(bytes.as_ptr().add(*offset).cast());
        let mut v1 = _mm_loadu_si128(bytes.as_ptr().add(*offset + 1).cast());
        let mut v2 = _mm_loadu_si128(bytes.as_ptr().add(*offset + 2).cast());
        let mut v3 = _mm_loadu_si128(bytes.as_ptr().add(*offset + 3).cast());

        let m0123s;
        if SHOULD_HASH {
            // We fused the unsigned -> signed conversion into the adds here.
            v0 = _mm_mullo_epi32(v0, vmul);
            v1 = _mm_mullo_epi32(v1, vmul);
            v2 = _mm_mullo_epi32(v2, vmul);
            v3 = _mm_mullo_epi32(v3, vmul);
            v0 = _mm_add_epi32(v0, vadd);
            v1 = _mm_add_epi32(v1, vadd);
            v2 = _mm_add_epi32(v2, vadd);
            v3 = _mm_add_epi32(v3, vadd);
            let m01s = _mm_min_epi32(v0, v1);
            let m23s = _mm_min_epi32(v2, v3);
            m0123s = _mm_min_epi32(m01s, m23s);
        } else {
            // We do a single conversion here after computing the unsigned minimum.
            let m01u = _mm_min_epu32(v0, v1);
            let m23u = _mm_min_epu32(v2, v3);
            let m0123u = _mm_min_epu32(m01u, m23u);
            m0123s = _mm_add_epi32(m0123u, _mm_set1_epi32(i32::MIN));
        }

        let voffset = _mm_set1_epi32(*offset as i32);
        let better = _mm_cmpgt_epi32(min_val, m0123s);
        min_val = _mm_min_epi32(min_val, m0123s);
        min_offset = _mm_blendv_epi8(min_offset, voffset, better);
        *offset += 16;
    };

    unsafe {
        while offset + 32 + 4 <= bytes.len() {
            // See note in avx512_impl: wrapping_add keeps the OOB prefetch
            // address computation sound.
            _mm_prefetch::<_MM_HINT_T0>(bytes.as_ptr().wrapping_add(offset + PREFETCH_DIST).cast());

            // Manually unrolled twice.
            body(&mut offset);
            body(&mut offset);
        }
        while offset + 4 <= bytes.len() {
            if offset + 16 + 3 > bytes.len() {
                offset = bytes.len() - 16 - 3;
            }

            body(&mut offset);
        }

        // Add the corresponding relative offsets for each 32-bit subregister
        // and interleave to 64-bit (value, offset) pairs.
        static SUBREG_OFFSET: [u32; 4] = [0, 4, 8, 12];
        min_offset = _mm_add_epi32(min_offset, _mm_loadu_si128(SUBREG_OFFSET.as_ptr().cast()));
        let pairs_lo = _mm_unpacklo_epi32(min_offset, min_val);
        let pairs_hi = _mm_unpackhi_epi32(min_offset, min_val);

        let mut out = [0i64; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), pairs_lo);
        _mm_storeu_si128(out.as_mut_ptr().add(2).cast(), pairs_hi);
        out.iter().copied().min().unwrap() as u32 as usize
    }
}

type ArgMinFn = unsafe fn(&[u8], u32, u32) -> usize;

static ARGMIN_IMPL: LazyLock<ArgMinFn> = LazyLock::new(|| {
    if is_x86_feature_detected!("avx512f") {
        avx512_impl::<false>
    } else if is_x86_feature_detected!("avx2") {
        avx2_impl::<false>
    } else if is_x86_feature_detected!("sse4.1") {
        sse41_impl::<false>
    } else {
        scalar::argmin_u32_overlapping_hashed::<false>
    }
});

static ARGMIN_HASH_IMPL: LazyLock<ArgMinFn> = LazyLock::new(|| {
    if is_x86_feature_detected!("avx512f") {
        avx512_impl::<true>
    } else if is_x86_feature_detected!("avx2") {
        avx2_impl::<true>
    } else if is_x86_feature_detected!("sse4.1") {
        sse41_impl::<true>
    } else {
        scalar::argmin_u32_overlapping_hashed::<true>
    }
});

pub fn argmin_u32_overlapping_hashed<const SHOULD_HASH: bool>(
    bytes: &[u8],
    multiplier: u32,
    addend: u32,
) -> usize {
    const MIN_SIMD_LEN: usize = 16 + 3;
    assert!(bytes.len() <= u32::MAX as usize);

    let within_four_offset = if SHOULD_HASH {
        unsafe { ARGMIN_HASH_IMPL(bytes, multiplier, addend) }
    } else {
        unsafe { ARGMIN_IMPL(bytes, multiplier, addend) }
    };

    if bytes.len() >= MIN_SIMD_LEN {
        let final_bump = scalar::argmin_u32_overlapping_hashed_four::<SHOULD_HASH>(
            &bytes[within_four_offset..],
            multiplier,
            addend,
        );
        within_four_offset + final_bump
    } else {
        within_four_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_ADDEND, DEFAULT_MULTIPLIER};
    use rand::distr::StandardUniform;
    use rand::prelude::*;

    // Reconstruct the full argmin from a "within-four" SIMD result, mirroring
    // the dispatch logic in argmin_u32_overlapping_hashed. The *_impl functions
    // only locate the block containing the global minimizer within the next four
    // bytes; the public wrapper resolves the exact position with a final scalar
    // bump, so the test must do the same before comparing to the scalar oracle.
    macro_rules! full_from_impl {
        ($within_four:expr, $hash:literal, $mul:expr, $add:expr, $bytes:expr) => {{
            const MIN_SIMD_LEN: usize = 16 + 3;
            let bytes: &[u8] = $bytes;
            let within_four = $within_four;
            if bytes.len() >= MIN_SIMD_LEN {
                within_four
                    + scalar::argmin_u32_overlapping_hashed_four::<$hash>(
                        &bytes[within_four..],
                        $mul,
                        $add,
                    )
            } else {
                within_four
            }
        }};
    }

    // Every SIMD width available on this CPU must produce exactly the same
    // argmin as the scalar oracle (and therefore as each other) for *all*
    // inputs. If two widths disagreed, two machines could place chunk
    // boundaries differently and silently break cross-machine deduplication.
    //
    // The existing `test_argmin_overlapped` only exercises the auto-dispatched
    // (best-available) width, leaving the narrower widths untested on wide-SIMD
    // hosts. This test calls each width directly, guarded by runtime detection.
    //
    // Note: under Rosetta 2 only the SSE4.1 path is reachable; AVX2/AVX-512 are
    // covered by CI on native x86 hardware.
    #[test]
    fn test_simd_widths_agree_with_scalar() {
        for size in 0..2048usize {
            let rng = SmallRng::seed_from_u64(size as u64);
            let bytes: Vec<u8> = rng.sample_iter(StandardUniform).take(size).collect();

            // SHOULD_HASH = false (multiplier/addend are ignored).
            let want = scalar::argmin_u32_overlapping_hashed::<false>(&bytes, 1, 0);
            if is_x86_feature_detected!("sse4.1") {
                let got = full_from_impl!(
                    unsafe { sse41_impl::<false>(&bytes, 1, 0) },
                    false,
                    1,
                    0,
                    &bytes
                );
                assert_eq!(got, want, "sse4.1 nohash size={size}");
            }
            if is_x86_feature_detected!("avx2") {
                let got = full_from_impl!(
                    unsafe { avx2_impl::<false>(&bytes, 1, 0) },
                    false,
                    1,
                    0,
                    &bytes
                );
                assert_eq!(got, want, "avx2 nohash size={size}");
            }
            if is_x86_feature_detected!("avx512f") {
                let got = full_from_impl!(
                    unsafe { avx512_impl::<false>(&bytes, 1, 0) },
                    false,
                    1,
                    0,
                    &bytes
                );
                assert_eq!(got, want, "avx512f nohash size={size}");
            }

            // SHOULD_HASH = true, using the default hash parameters.
            let (mul, add) = (DEFAULT_MULTIPLIER, DEFAULT_ADDEND);
            let want = scalar::argmin_u32_overlapping_hashed::<true>(&bytes, mul, add);
            if is_x86_feature_detected!("sse4.1") {
                let got = full_from_impl!(
                    unsafe { sse41_impl::<true>(&bytes, mul, add) },
                    true,
                    mul,
                    add,
                    &bytes
                );
                assert_eq!(got, want, "sse4.1 hash size={size}");
            }
            if is_x86_feature_detected!("avx2") {
                let got = full_from_impl!(
                    unsafe { avx2_impl::<true>(&bytes, mul, add) },
                    true,
                    mul,
                    add,
                    &bytes
                );
                assert_eq!(got, want, "avx2 hash size={size}");
            }
            if is_x86_feature_detected!("avx512f") {
                let got = full_from_impl!(
                    unsafe { avx512_impl::<true>(&bytes, mul, add) },
                    true,
                    mul,
                    add,
                    &bytes
                );
                assert_eq!(got, want, "avx512f hash size={size}");
            }
        }
    }
}
