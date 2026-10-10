//! Named random streams (ADR 0021). A stream's draws depend on the run's seed and
//! the stream's name alone: its key is the state NumPy's
//! `SeedSequence(seed, spawn_key=name)` generates, which is how the Python side
//! names its own streams, so one seed names the same streams in both, and its
//! draws are ChaCha20's, the same bits on every machine.

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const POOL: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_L: u32 = 0xca01_f9dd;
const MIX_R: u32 = 0x4973_f715;

fn hashmix(value: u32, constant: &mut u32) -> u32 {
    let value = value ^ *constant;
    *constant = constant.wrapping_mul(MULT_A);
    let value = value.wrapping_mul(*constant);
    value ^ (value >> 16)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
    result ^ (result >> 16)
}

/// The `words` that NumPy's `SeedSequence(seed, spawn_key=key)` generates, by its
/// own algorithm: the seed as little-endian 32-bit words, as many as it needs and
/// at least one, padded with zeros to the pool where a key follows, hashed into a
/// pool of four and drawn out of it.
pub fn seed_state(seed: u128, key: &[u32], words: usize) -> Vec<u32> {
    let mut entropy = Vec::with_capacity(POOL + key.len());
    let mut rest = seed;
    loop {
        entropy.push(rest as u32);
        rest >>= 32;
        if rest == 0 {
            break;
        }
    }
    if !key.is_empty() && entropy.len() < POOL {
        entropy.resize(POOL, 0);
    }
    entropy.extend_from_slice(key);
    let mut constant = INIT_A;
    let mut pool = [0u32; POOL];
    for (i, word) in pool.iter_mut().enumerate() {
        *word = hashmix(entropy.get(i).copied().unwrap_or(0), &mut constant);
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &mut constant));
            }
        }
    }
    for &word in entropy.iter().skip(POOL) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(word, &mut constant));
        }
    }
    let mut constant = INIT_B;
    (0..words)
        .map(|i| {
            let value = pool[i % POOL] ^ constant;
            constant = constant.wrapping_mul(MULT_B);
            let value = value.wrapping_mul(constant);
            value ^ (value >> 16)
        })
        .collect()
}

/// The key a stream's `name` spells, a word for each of its bytes, as Python's
/// streams spell theirs.
pub fn named(name: &str) -> Vec<u32> {
    name.bytes().map(u32::from).collect()
}

/// The stream `key` names in a run seeded with `seed`: ChaCha20, keyed by the
/// eight words the two derive.
pub fn stream(seed: u128, key: &[u32]) -> ChaCha20Rng {
    let mut bytes = [0u8; 32];
    for (chunk, word) in bytes.chunks_exact_mut(4).zip(seed_state(seed, key, 8)) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    ChaCha20Rng::from_seed(bytes)
}

#[cfg(test)]
mod tests {
    use super::{named, seed_state};

    #[test]
    fn a_seed_and_a_name_give_the_state_numpy_gives() {
        // numpy 2.5.3: SeedSequence(seed, spawn_key=key).generate_state(8)
        let intrabar = named("intrabar");
        let mut third = intrabar.clone();
        third.push(3);
        let mut fifth = intrabar.clone();
        fifth.push(5);
        let cases: [(u128, Vec<u32>, [u32; 8]); 7] = [
            (
                42,
                intrabar.clone(),
                [
                    772_974_723,
                    816_912_719,
                    1_177_258_273,
                    3_894_579_219,
                    1_665_282_931,
                    2_520_140_088,
                    4_048_224_945,
                    3_913_466_419,
                ],
            ),
            (
                0,
                intrabar,
                [
                    2_948_267_288,
                    3_594_707_689,
                    856_073_469,
                    1_361_612_106,
                    3_214_479_947,
                    586_392_888,
                    1_102_330_223,
                    1_581_090_159,
                ],
            ),
            (
                (1 << 40) + 7,
                third,
                [
                    430_806_972,
                    880_068_412,
                    185_498_581,
                    2_418_091_963,
                    1_003_254_909,
                    1_897_447_392,
                    478_994_743,
                    3_532_678_810,
                ],
            ),
            (
                7,
                Vec::new(),
                [
                    2_083_679_832,
                    3_939_563_265,
                    4_185_785_210,
                    142_198_000,
                    1_732_601_695,
                    1_648_004_171,
                    1_614_310_572,
                    3_007_594_413,
                ],
            ),
            (
                0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
                fifth,
                [
                    3_091_813_313,
                    424_042_181,
                    4_141_472_470,
                    2_041_048_034,
                    2_720_235_611,
                    1_219_033_227,
                    2_522_109_851,
                    3_357_033_181,
                ],
            ),
            (
                (1 << 127) + 3,
                Vec::new(),
                [
                    1_500_925_730,
                    2_799_241_227,
                    1_464_605_349,
                    2_015_901_335,
                    1_171_665_266,
                    2_149_327_678,
                    2_793_893_710,
                    1_843_789_114,
                ],
            ),
            (
                1 << 96,
                named("intrabar"),
                [
                    2_611_999_141,
                    3_258_249_127,
                    430_883_520,
                    3_696_726_673,
                    53_748_046,
                    1_398_505_672,
                    752_794_816,
                    4_219_388_712,
                ],
            ),
        ];
        for (seed, key, numpy) in cases {
            assert_eq!(seed_state(seed, &key, 8), numpy, "seed {seed}");
        }
    }
}
