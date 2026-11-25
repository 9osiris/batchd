#include "model.h"

namespace batchd {

// splitmix64, the usual cheap hash. deterministic across runs.
static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint32_t next_token(uint64_t seq_id, int position) {
    uint64_t h = mix(seq_id * 0x100000001b3ULL ^ (uint64_t)(position + 1) * 0x9e3779b9ULL);
    return (uint32_t)(h % 32000);
}

bool is_eos(uint32_t tok) {
    return tok == 2;
}

}  // namespace batchd
