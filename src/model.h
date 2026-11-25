#pragma once

#include <cstdint>

namespace batchd {

// deterministic fake model: given a sequence id and a position it
// always returns the same token. no weights, no real inference, just
// something stable enough to drive the scheduler and its tests.
uint32_t next_token(uint64_t seq_id, int position);

// token 2 is end of sequence, same as a lot of real tokenizers.
bool is_eos(uint32_t tok);

}  // namespace batchd
