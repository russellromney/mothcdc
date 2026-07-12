pub fn argmin_u32_overlapping_hashed<const SHOULD_HASH: bool>(
    bytes: &[u8],
    multiplier: u32,
    addend: u32,
) -> usize {
    let mut min_idx = 0;
    let mut min_val = u32::MAX;

    if bytes.len() < 4 {
        return 0;
    }

    for (i, window) in bytes.windows(4).enumerate() {
        let mut v = u32::from_le_bytes(window.try_into().unwrap());
        if SHOULD_HASH {
            v = v.wrapping_mul(multiplier);
            v = v.wrapping_add(addend);
        }
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }

    min_idx
}

/// Length of the common prefix of `a` and `b` (compared over the shorter one).
///
/// This is the portable reference for the packed-scanning primitive used by the
/// caterpillar fast path: `common_prefix_len(&data[s + u..], &data[s..])` is
/// exactly how far `data` stays periodic with period `u` from position `s`.
/// Word-at-a-time rather than byte-at-a-time so even the fallback is not a
/// scalar byte loop.
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return i + (diff.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Length of the prefix of `data` consisting entirely of `byte`.
///
/// Portable reference for the broadcast variant of packed scanning (the
/// caterpillar fast path for constant-byte runs like zero-fill).
pub fn byte_run_len(data: &[u8], byte: u8) -> usize {
    let rep = u64::from_le_bytes([byte; 8]);
    let n = data.len();
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
        let diff = x ^ rep;
        if diff != 0 {
            return i + (diff.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && data[i] == byte {
        i += 1;
    }
    i
}

#[inline(always)]
#[allow(dead_code)]
pub fn argmin_u32_overlapping_hashed_four<const SHOULD_HASH: bool>(
    bytes: &[u8],
    multiplier: u32,
    addend: u32,
) -> usize {
    assert!(bytes.len() >= 7);
    (0..4)
        .min_by_key(|i| {
            let substr = &bytes[*i..*i + 4];
            let mut v = u32::from_le_bytes(substr.try_into().unwrap());
            if SHOULD_HASH {
                v = v.wrapping_mul(multiplier);
                v = v.wrapping_add(addend);
            }
            v
        })
        .unwrap()
}
