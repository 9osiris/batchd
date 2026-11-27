#include "sched.h"

#include "model.h"

#include <algorithm>

namespace batchd {

Scheduler::Scheduler(Config cfg) : cfg_(cfg) {
    if (cfg_.max_batch_tokens < 1) cfg_.max_batch_tokens = 1;
    if (cfg_.max_seqs < 1) cfg_.max_seqs = 1;
    if (cfg_.block_size < 1) cfg_.block_size = 1;
    if (cfg_.max_blocks < 1) cfg_.max_blocks = 1;
    if (cfg_.default_max_tokens < 1) cfg_.default_max_tokens = 1;
}

uint64_t Scheduler::submit(int prompt_len, int max_tokens, int prefix_len,
                         uint64_t prefix_id) {
    Sequence s;
    s.id = next_id_++;
    s.prompt_len = prompt_len < 1 ? 1 : prompt_len;
    s.max_tokens = max_tokens < 0 ? cfg_.default_max_tokens : max_tokens;
    if (s.max_tokens < 1) s.max_tokens = 1;
    s.prefix_len = prefix_len < 0 ? 0 : prefix_len;
    if (s.prefix_len > s.prompt_len) s.prefix_len = s.prompt_len;
    s.prefix_id = prefix_id;
    s.arrival_step = step_;
    seqs_.push_back(s);
    return s.id;
}

int Scheduler::blocks_for(int tokens) const {
    return (tokens + cfg_.block_size - 1) / cfg_.block_size;
}

int Scheduler::running_count() const {
    int n = 0;
    for (const auto& s : seqs_)
        if (s.running && !s.done) n++;
    return n;
}

int Scheduler::waiting_count() const {
    int n = 0;
    for (const auto& s : seqs_)
        if (!s.running && !s.done) n++;
    return n;
}

bool Scheduler::finished() const {
    for (const auto& s : seqs_)
        if (!s.done) return false;
    return true;
}

int Scheduler::shared_tokens_for(const Sequence& s) const {
    if (s.prefix_id == 0 || s.prefix_len <= 0) return 0;
    return (s.prefix_len / cfg_.block_size) * cfg_.block_size;
}

bool Scheduler::try_attach(Sequence& s) {
    if (s.attached || s.prefix_id == 0) return false;
    int shared = shared_tokens_for(s);
    if (shared <= 0) return false;
    auto it = prefixes_.find({s.prefix_id, shared});
    if (it == prefixes_.end()) return false;
    if (s.prefilled >= shared) {
        // already computed the prefix, free our own copy of its blocks
        used_blocks_ -= it->second.blocks;
        s.blocks -= it->second.blocks;
    } else {
        // skip computing the cached prefix, drop any partial copy we hold
        used_blocks_ -= s.blocks;
        s.blocks = 0;
        prefix_tokens_saved_ += shared - s.prefilled;
        s.prefilled = shared;
    }
    s.shared_blocks = it->second.blocks;
    s.attached = true;
    it->second.refs++;
    prefix_hits_++;
    return true;
}

void Scheduler::register_prefix(Sequence& s) {
    if (s.attached || s.prefix_id == 0) return;
    int shared = shared_tokens_for(s);
    if (shared <= 0 || s.prefilled < shared) return;
    if (prefixes_.count({s.prefix_id, shared})) return;  // owned already
    PrefixEntry e;
    e.blocks = blocks_for(shared);
    e.refs = 1;
    prefixes_[{s.prefix_id, shared}] = e;
    s.blocks -= e.blocks;
    s.shared_blocks = e.blocks;
    s.attached = true;
}

void Scheduler::release_prefix(Sequence& s) {
    if (!s.attached) return;
    auto it = prefixes_.find({s.prefix_id, shared_tokens_for(s)});
    if (it != prefixes_.end()) {
        if (--it->second.refs <= 0) {
            used_blocks_ -= it->second.blocks;
            prefixes_.erase(it);
        }
    }
    s.shared_blocks = 0;
    s.attached = false;
}

void Scheduler::free_seq(Sequence& s) {
    release_prefix(s);
    used_blocks_ -= s.blocks;
    s.blocks = 0;
}

void Scheduler::finish_seq(Sequence& s) {
    s.done = true;
    s.running = false;
    s.finish_step = step_;
    free_seq(s);
}

bool Scheduler::preempt_youngest(Sequence* except) {
    Sequence* victim = nullptr;
    for (auto& s : seqs_) {
        if (!s.running || s.done) continue;
        if (except && s.id == except->id) continue;
        if (!victim || s.id > victim->id) victim = &s;
    }
    if (!victim) return false;
    free_seq(*victim);
    victim->running = false;
    victim->prefilled = 0;
    victim->generated = 0;
    victim->first_token_step = -1;
    victim->preemptions++;
    preemptions_++;
    return true;
}

void Scheduler::step() {
    int budget = cfg_.max_batch_tokens;
    int step_tokens = 0;

    // admit waiting sequences while there is room in the batch.
    for (auto& s : seqs_) {
        if (s.done || s.running) continue;
        if (running_count() >= cfg_.max_seqs) break;
        s.running = true;
        // a cached prefix means the prompt is already partly computed
        try_attach(s);
    }
    if (running_count() > peak_running_) peak_running_ = running_count();

    // prefill pass, in submit order. a sequence that finishes prefill
    // this step emits its first token right away.
    for (auto& s : seqs_) {
        if (!s.running || s.done) continue;
        // the prefix may have been cached since this sequence was
        // admitted, by a sequence ahead of it in submit order.
        try_attach(s);
        if (s.prefilled >= s.prompt_len) {
            if (s.generated == 0) {
                // whole prompt was cached, first token costs no compute
                uint32_t tok = next_token(s.id, 0);
                s.generated = 1;
                s.first_token_step = step_;
                if (is_eos(tok) || s.generated >= s.max_tokens) finish_seq(s);
            }
            continue;
        }
        if (budget <= 0) break;
        int remaining = s.prompt_len - s.prefilled;
        int take = remaining;
        if (cfg_.chunked_prefill) {
            take = std::min(take, budget);
        } else if (take > budget) {
            continue;  // whole prompt or nothing, wait for a roomier step
        }
        int need = blocks_for(s.prefilled + take) - s.blocks - s.shared_blocks;
        // prefill never preempts: if the blocks are not free, this
        // sequence waits for a roomier step. preempting here livelocks,
        // the newcomer would evict the seq it just waited behind and
        // get evicted itself on the next step.
        if (used_blocks_ + need > cfg_.max_blocks) continue;
        used_blocks_ += need;
        s.blocks += need;
        s.prefilled += take;
        budget -= take;
        step_tokens += take;
        prefill_tokens_ += take;
        // crossing the shared prefix registers it for later sequences,
        // or attaches to one another sequence registered first.
        if (!s.attached) register_prefix(s);
        if (s.prefilled == s.prompt_len) {
            uint32_t tok = next_token(s.id, 0);
            s.generated = 1;
            s.first_token_step = step_;
            if (is_eos(tok) || s.generated >= s.max_tokens) finish_seq(s);
        }
    }

    // decode pass, one token per running sequence that already
    // prefilled. sequences that just finished prefill this step
    // already got their first token, they sit this one out.
    for (auto& s : seqs_) {
        if (!s.running || s.done) continue;
        if (s.prefilled < s.prompt_len) continue;
        if (s.first_token_step == step_) continue;
        if (budget < 1) break;
        int need =
            blocks_for(s.prefilled + s.generated + 1) - s.blocks - s.shared_blocks;
        while (need > 0 && used_blocks_ + need > cfg_.max_blocks) {
            if (!preempt_youngest(&s)) break;
        }
        if (used_blocks_ + need > cfg_.max_blocks) continue;
        used_blocks_ += need;
        s.blocks += need;
        uint32_t tok = next_token(s.id, s.generated);
        s.generated++;
        budget -= 1;
        step_tokens += 1;
        decode_tokens_ += 1;
        if (is_eos(tok) || s.generated >= s.max_tokens) finish_seq(s);
    }

    last_step_tokens_ = step_tokens;
    step_++;
}

void Scheduler::run() {
    // a config that can never make progress (chunked prefill off and
    // the budget smaller than the longest prompt, or blocks too small
    // to hold one prompt) would loop forever, so stop after a while.
    const int cap = 100000;
    while (!finished() && step_ < cap) step();
}

Metrics Scheduler::metrics() const {
    Metrics m;
    m.steps = step_;
    m.prefill_tokens = prefill_tokens_;
    m.decode_tokens = decode_tokens_;
    m.preemptions = preemptions_;
    m.peak_running = peak_running_;
    m.prefix_hits = prefix_hits_;
    m.prefix_tokens_saved = prefix_tokens_saved_;
    double ttft = 0, tpot = 0;
    int n_ttft = 0, n_tpot = 0;
    for (const auto& s : seqs_) {
        if (s.first_token_step < 0) continue;
        ttft += (double)(s.first_token_step - s.arrival_step);
        n_ttft++;
        if (s.generated > 1 && s.finish_step >= 0) {
            tpot += (double)(s.finish_step - s.first_token_step) / (s.generated - 1);
            n_tpot++;
        }
    }
    if (n_ttft) m.mean_ttft = ttft / n_ttft;
    if (n_tpot) m.mean_tpot = tpot / n_tpot;
    return m;
}

}  // namespace batchd
