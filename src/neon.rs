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
