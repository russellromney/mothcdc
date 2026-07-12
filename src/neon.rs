use crate::scalar;

pub fn argmin_u32_overlapping_hashed<const SHOULD_HASH: bool>(
    bytes: &[u8],
    multiplier: u32,
    addend: u32,
) -> usize {
    use std::arch::aarch64::*;

    assert!(bytes.len() < u32::MAX as usize);

    if bytes.len() < 16 + 3 {
        return scalar::argmin_u32_overlapping_hashed::<SHOULD_HASH>(bytes, multiplier, addend);
    }

    // Fast path for chunk starting with zeros (relatively common).
    if !SHOULD_HASH && bytes[0..4] == [0, 0, 0, 0] {
        return 0;
    }

    unsafe {
        // For i in 0..4 we have that:
        //   min_val[i] will contain the minimum of all possible bytes[16*k + 4*i..16*k + 4*(i+1)].
        //   min_offset[i] will contain the offset 16*k at which this occurs.
        // This means after the vectorized loop we only have to choose the smallest min_val[i]
        // and check four positions min_offset[i]..min_offset[i] + 4 for the overall minimum.
        let mut min_val = vdupq_n_u32(u32::MAX);
        let mut min_offset = vdupq_n_u32(0);

        let mut offset = 0;
        let vmul = vdupq_n_u32(multiplier);
        let vadd = vdupq_n_u32(addend);

        let mut body = |offset: &mut usize| {
            let mut v0 = vld1q_u32(bytes.as_ptr().add(*offset).cast());
            let mut v1 = vld1q_u32(bytes.as_ptr().add(*offset + 1).cast());
            let mut v2 = vld1q_u32(bytes.as_ptr().add(*offset + 2).cast());
            let mut v3 = vld1q_u32(bytes.as_ptr().add(*offset + 3).cast());

            if SHOULD_HASH {
                v0 = vmlaq_u32(vadd, vmul, v0);
                v1 = vmlaq_u32(vadd, vmul, v1);
                v2 = vmlaq_u32(vadd, vmul, v2);
                v3 = vmlaq_u32(vadd, vmul, v3);
            }

            let m01 = vminq_u32(v0, v1);
            let m23 = vminq_u32(v2, v3);
            let m0123 = vminq_u32(m01, m23);

            let better = vcltq_u32(m0123, min_val);
            min_offset = vbslq_u32(better, vdupq_n_u32(*offset as u32), min_offset);
            min_val = vminq_u32(m0123, min_val);
            *offset += 16;
        };

        while offset + 32 + 4 <= bytes.len() {
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

        let mut min_val_scalar = [0u32; 4];
        let mut min_offset_scalar = [0u32; 4];
        vst1q_u32(min_val_scalar.as_mut_ptr(), min_val);
        vst1q_u32(min_offset_scalar.as_mut_ptr(), min_offset);

        let min_reg = (0..4)
            .map(|i| {
                ((min_val_scalar[i] as u64) << 32) | (min_offset_scalar[i] as u64 + 4 * i as u64)
            })
            .min()
            .unwrap();
        let min_offset_base = min_reg as u32 as usize;

        let min_offset_inc = scalar::argmin_u32_overlapping_hashed_four::<SHOULD_HASH>(
            bytes.get_unchecked(min_offset_base..),
            multiplier,
            addend,
        );
        min_offset_base + min_offset_inc
    }
}

// ---------------------------------------------------------------------------
// Packed scanning (VectorCDC-style) primitives for the caterpillar fast path.
// See src/x86_64.rs for the full description; the NEON versions differ only in
// how the "all lanes equal?" test and mismatch position are extracted, since
// NEON has no movemask: `vminvq_u8` of the comparison is 0xFF iff every lane
// matched, and on a mismatch `vshrn` narrows the comparison to a 64-bit mask
// with 4 bits per byte, whose trailing ones locate the first differing byte.
// ---------------------------------------------------------------------------

/// Index of the first lane of `eq` (a `vceqq_u8` result) that is not all-ones.
/// Must only be called when at least one lane mismatched.
#[inline(always)]
unsafe fn first_mismatch_lane(eq: std::arch::aarch64::uint8x16_t) -> usize {
    use std::arch::aarch64::*;
    unsafe {
        let nib = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
        let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(nib));
        (mask.trailing_ones() / 4) as usize
    }
}

/// Length of the common prefix of `a` and `b` (compared over the shorter one).
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    use std::arch::aarch64::*;

    let n = a.len().min(b.len());
    let mut i = 0;
    unsafe {
        // Skip 64 B per iteration while everything matches; the first block
        // with a mismatch falls through to the single-vector loop to locate it.
        while i + 64 <= n {
            let e0 = vceqq_u8(vld1q_u8(a.as_ptr().add(i)), vld1q_u8(b.as_ptr().add(i)));
            let e1 = vceqq_u8(
                vld1q_u8(a.as_ptr().add(i + 16)),
                vld1q_u8(b.as_ptr().add(i + 16)),
            );
            let e2 = vceqq_u8(
                vld1q_u8(a.as_ptr().add(i + 32)),
                vld1q_u8(b.as_ptr().add(i + 32)),
            );
            let e3 = vceqq_u8(
                vld1q_u8(a.as_ptr().add(i + 48)),
                vld1q_u8(b.as_ptr().add(i + 48)),
            );
            let all = vandq_u8(vandq_u8(e0, e1), vandq_u8(e2, e3));
            if vminvq_u8(all) != 0xFF {
                break;
            }
            i += 64;
        }
        while i + 16 <= n {
            let va = vld1q_u8(a.as_ptr().add(i));
            let vb = vld1q_u8(b.as_ptr().add(i));
            let eq = vceqq_u8(va, vb);
            if vminvq_u8(eq) != 0xFF {
                return i + first_mismatch_lane(eq);
            }
            i += 16;
        }
    }
    i + scalar::common_prefix_len(&a[i..n], &b[i..n])
}

/// Length of the prefix of `data` consisting entirely of `byte`.
pub fn byte_run_len(data: &[u8], byte: u8) -> usize {
    use std::arch::aarch64::*;

    let n = data.len();
    let mut i = 0;
    unsafe {
        let needle = vdupq_n_u8(byte);
        // 64 B skip loop; see common_prefix_len.
        while i + 64 <= n {
            let e0 = vceqq_u8(vld1q_u8(data.as_ptr().add(i)), needle);
            let e1 = vceqq_u8(vld1q_u8(data.as_ptr().add(i + 16)), needle);
            let e2 = vceqq_u8(vld1q_u8(data.as_ptr().add(i + 32)), needle);
            let e3 = vceqq_u8(vld1q_u8(data.as_ptr().add(i + 48)), needle);
            let all = vandq_u8(vandq_u8(e0, e1), vandq_u8(e2, e3));
            if vminvq_u8(all) != 0xFF {
                break;
            }
            i += 64;
        }
        while i + 16 <= n {
            let v = vld1q_u8(data.as_ptr().add(i));
            let eq = vceqq_u8(v, needle);
            if vminvq_u8(eq) != 0xFF {
                return i + first_mismatch_lane(eq);
            }
            i += 16;
        }
    }
    i + scalar::byte_run_len(&data[i..], byte)
}
