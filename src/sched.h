#pragma once

#include <cstdint>
#include <vector>

namespace batchd {

struct Config {
    int max_batch_tokens = 512;  // token budget per step, prefill + decode
    int max_seqs = 32;           // most sequences running at once
    int block_size = 16;         // tokens per kv block
    int max_blocks = 256;        // total kv blocks, the memory ceiling
    bool chunked_prefill = true; // split long prefills across steps
    int default_max_tokens = 32; // generated tokens per sequence
};

struct Sequence {
    uint64_t id = 0;
    int prompt_len = 0;
    int max_tokens = 32;
    int prefilled = 0;
    int generated = 0;
    int blocks = 0;
    int arrival_step = 0;
    int first_token_step = -1;
    int finish_step = -1;
    int preemptions = 0;
    bool done = false;
    bool running = false;
};

struct Metrics {
    int steps = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int preemptions = 0;
    int peak_running = 0;
    double mean_ttft = 0;  // steps from arrival to first token
    double mean_tpot = 0;  // steps per output token after the first
};

// continuous batching scheduler. sequences arrive, get admitted while
// there is room, prefill in chunks against a per-step token budget,
// then decode one token per step each. kv blocks are the memory: when
// a decoding sequence needs a block and none is free, the youngest
// other running sequence is preempted, goes back to waiting, and
// recomputes from scratch when it is admitted again. prefill never
// preempts, a sequence that cannot fit waits for a roomier step.
class Scheduler {
public:
    explicit Scheduler(Config cfg);

    // queue a sequence, returns its id. arrival is the current step.
    uint64_t submit(int prompt_len, int max_tokens = -1);

    void step();       // one scheduling step
    void run();        // steps until finished, capped so a stuck config stops
    bool finished() const;

    const std::vector<Sequence>& sequences() const { return seqs_; }
    Metrics metrics() const;
    int step_count() const { return step_; }
    int used_blocks() const { return used_blocks_; }
    int running_count() const;
    int waiting_count() const;
    int last_step_tokens() const { return last_step_tokens_; }

private:
    Config cfg_;
    std::vector<Sequence> seqs_;
    uint64_t next_id_ = 1;
    int step_ = 0;
    int used_blocks_ = 0;
    int preemptions_ = 0;
    int peak_running_ = 0;
    int prefill_tokens_ = 0;
    int decode_tokens_ = 0;
    int last_step_tokens_ = 0;

    int blocks_for(int tokens) const;
    void free_seq(Sequence& s);
    void finish_seq(Sequence& s);
    // preempt the youngest running sequence other than except.
    // returns false when there is nothing to preempt.
    bool preempt_youngest(Sequence* except);
};

}  // namespace batchd
