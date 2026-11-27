#pragma once

#include <cstdint>
#include <map>
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
    int blocks = 0;         // own kv blocks, not counting shared ones
    int shared_blocks = 0;  // blocks borrowed from the prefix cache
    int prefix_len = 0;     // leading tokens shared with other sequences
    uint64_t prefix_id = 0; // 0 means no shared prefix
    bool attached = false;  // currently holding shared prefix blocks
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
    int prefix_hits = 0;        // sequences that attached to a cached prefix
    int prefix_tokens_saved = 0; // prompt tokens skipped via the cache
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
    // prefix_len/prefix_id declare a shared prompt start: sequences
    // with the same id share the first prefix_len tokens, so the kv
    // blocks for the whole blocks inside that prefix are allocated
    // once and the later sequences skip computing them.
    uint64_t submit(int prompt_len, int max_tokens = -1, int prefix_len = 0,
                    uint64_t prefix_id = 0);

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
    // one cached prefix: the blocks for a shared prompt start, held
    // once no matter how many sequences use them. refs counts the
    // sequences attached right now; the entry frees when it hits 0.
    // keyed by (prefix id, shared length) so different-length prefixes
    // under one id are different entries.
    struct PrefixEntry {
        int blocks = 0;
        int refs = 0;
    };

    Config cfg_;
    std::vector<Sequence> seqs_;
    std::map<std::pair<uint64_t, int>, PrefixEntry> prefixes_;
    uint64_t next_id_ = 1;
    int step_ = 0;
    int used_blocks_ = 0;
    int preemptions_ = 0;
    int peak_running_ = 0;
    int prefill_tokens_ = 0;
    int decode_tokens_ = 0;
    int last_step_tokens_ = 0;
    int prefix_hits_ = 0;
    int prefix_tokens_saved_ = 0;

    int blocks_for(int tokens) const;
    // whole-block tokens inside a sequence's declared prefix. the
    // tail of a prefix that does not fill a block stays private.
    int shared_tokens_for(const Sequence& s) const;
    // attach a sequence to a cached prefix, if one matches. when the
    // sequence has not prefilled yet it skips the shared tokens; when
    // it already computed them its own copy of those blocks is freed.
    bool try_attach(Sequence& s);
    // register the blocks a sequence just prefilled as a cached
    // prefix, so later sequences with the same id can attach to them.
    void register_prefix(Sequence& s);
    // drop a sequence's hold on a cached prefix, freeing the entry
    // when the last sequence lets go.
    void release_prefix(Sequence& s);
    void free_seq(Sequence& s);
    void finish_seq(Sequence& s);
    // preempt the youngest running sequence other than except.
    // returns false when there is nothing to preempt.
    bool preempt_youngest(Sequence* except);
};

}  // namespace batchd
