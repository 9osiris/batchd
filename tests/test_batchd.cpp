#include "../src/model.h"
#include "../src/sched.h"

#include <cstdio>
#include <vector>

using namespace batchd;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        passed++;
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

static const Sequence* find(const Scheduler& sch, uint64_t id) {
    for (const auto& s : sch.sequences())
        if (s.id == id) return &s;
    return nullptr;
}

int main() {
    // model: deterministic, eos is rare
    check(next_token(1, 0) == next_token(1, 0), "model deterministic");
    check(next_token(1, 0) != next_token(2, 0), "model varies by seq");
    check(next_token(1, 0) != next_token(1, 1), "model varies by position");
    check(is_eos(2) && !is_eos(3), "eos is token 2");
    int eos_seen = 0;
    for (int i = 0; i < 1000000; i++)
        if (is_eos(next_token(9, i))) eos_seen++;
    check(eos_seen > 0 && eos_seen < 200, "eos rare but reachable");

    // single sequence runs to its max tokens and frees its blocks
    {
        Config cfg;
        Scheduler sch(cfg);
        uint64_t id = sch.submit(32, 8);
        sch.run();
        const Sequence* s = find(sch, id);
        check(s && s->done, "single seq finishes");
        check(s && s->generated == 8, "single seq generates max tokens");
        check(s && s->first_token_step == 0, "single seq ttft is 0");
        check(sch.used_blocks() == 0, "blocks freed after finish");
        check(sch.metrics().steps == 8, "single seq takes 8 steps");
    }

    // chunked prefill: prompt 20, budget 8, first token on step 2
    {
        Config cfg;
        cfg.max_batch_tokens = 8;
        Scheduler sch(cfg);
        uint64_t id = sch.submit(20, 4);
        sch.step();
        check(find(sch, id)->prefilled == 8, "chunk 1 prefills 8");
        check(find(sch, id)->first_token_step == -1, "no token after chunk 1");
        sch.step();
        check(find(sch, id)->prefilled == 16, "chunk 2 prefills 8");
        sch.step();
        const Sequence* s = find(sch, id);
        check(s->prefilled == 20 && s->first_token_step == 2,
              "chunked prefill first token on step 2");
    }

    // no chunked prefill: whole prompt or nothing
    {
        Config cfg;
        cfg.max_batch_tokens = 32;
        cfg.chunked_prefill = false;
        Scheduler sch(cfg);
        uint64_t id = sch.submit(20, 4);
        sch.step();
        check(find(sch, id)->first_token_step == 0,
              "no-chunk prefill fits in one step");
    }
    {
        Config cfg;
        cfg.max_batch_tokens = 8;
        cfg.chunked_prefill = false;
        Scheduler sch(cfg);
        uint64_t id = sch.submit(20, 4);
        for (int i = 0; i < 3; i++) sch.step();
        const Sequence* s = find(sch, id);
        check(s->prefilled == 0 && s->first_token_step == -1,
              "no-chunk prefill waits when budget is too small");
        check(!sch.finished(), "stuck config never finishes");
    }

    // decode: one token per step per running sequence
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        uint64_t a = sch.submit(4, 5);
        uint64_t b = sch.submit(4, 5);
        sch.step();  // both prefill and emit token 1
        check(find(sch, a)->generated == 1 && find(sch, b)->generated == 1,
              "both seqs emit first token on step 0");
        sch.step();
        check(find(sch, a)->generated == 2 && find(sch, b)->generated == 2,
              "both seqs decode one token on step 1");
        sch.run();
        check(find(sch, a)->generated == 5 && find(sch, b)->generated == 5,
              "both seqs reach max tokens");
    }

    // per-step token budget is never exceeded
    {
        Config cfg;
        cfg.max_batch_tokens = 16;
        Scheduler sch(cfg);
        for (int i = 0; i < 6; i++) sch.submit(50, 10);
        while (!sch.finished()) {
            sch.step();
            check(sch.last_step_tokens() <= 16, "step stays in budget");
        }
    }

    // max_seqs caps the running batch
    {
        Config cfg;
        cfg.max_seqs = 2;
        Scheduler sch(cfg);
        for (int i = 0; i < 6; i++) sch.submit(16, 4);
        while (!sch.finished()) {
            sch.step();
            check(sch.running_count() <= 2, "running batch capped at 2");
        }
        check(sch.metrics().peak_running == 2, "peak batch hits the cap");
    }

    // kv blocks: prompt 40 at block 16 needs 3 blocks, freed on finish
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        uint64_t id = sch.submit(40, 2);
        sch.step();
        check(find(sch, id)->blocks == 3, "40 tokens take 3 blocks");
        check(sch.used_blocks() == 3, "used blocks counts the seq");
        sch.run();
        check(sch.used_blocks() == 0, "blocks freed when seq finishes");
    }

    // preemption: two seqs that both fit after prefill, but whose
    // decode growth outgrows kv memory. the preempted one recomputes
    // and still finishes.
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        cfg.max_blocks = 4;
        cfg.block_size = 16;
        Scheduler sch(cfg);
        uint64_t a = sch.submit(32, 32);
        uint64_t b = sch.submit(32, 32);
        sch.run();
        check(sch.finished(), "preempted workload still finishes");
        check(sch.metrics().preemptions >= 1, "at least one preemption");
        check(find(sch, a)->done && find(sch, b)->done,
              "both seqs done after preemption");
        check(find(sch, a)->preemptions + find(sch, b)->preemptions >= 1,
              "preemption counted on a sequence");
        check(sch.used_blocks() == 0, "blocks freed after preemptions");
    }

    // prefill waits instead of preempting: two long seqs, only room
    // for one prompt at a time. no preemption, both still finish.
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        cfg.max_blocks = 4;
        cfg.block_size = 16;
        Scheduler sch(cfg);
        uint64_t a = sch.submit(48, 6);
        uint64_t b = sch.submit(48, 6);
        sch.run();
        check(sch.finished(), "prefill-wait workload finishes");
        check(sch.metrics().preemptions == 0, "prefill never preempts");
        check(find(sch, a)->done && find(sch, b)->done,
              "both seqs done after waiting");
    }

    // determinism: same workload twice, same finish steps
    {
        auto run_once = []() {
            Config cfg;
            cfg.max_batch_tokens = 32;
            cfg.max_blocks = 16;
            Scheduler sch(cfg);
            for (int i = 0; i < 8; i++) sch.submit(20 + i * 7, 10);
            sch.run();
            std::vector<int> out;
            for (const auto& s : sch.sequences()) out.push_back(s.finish_step);
            return out;
        };
        check(run_once() == run_once(), "same workload, same finish steps");
    }

    // metrics: ttft and tpot on a known workload
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(8, 4);
        sch.run();
        Metrics m = sch.metrics();
        check(m.steps == 4, "metrics steps");
        check(m.mean_ttft == 0, "metrics ttft 0 for instant prefill");
        check(m.mean_tpot == 1, "metrics tpot 1 step per token");
        check(m.prefill_tokens == 8 && m.decode_tokens == 3,
              "metrics split prefill and decode");
    }

    // run() stops instead of spinning on a stuck config
    {
        Config cfg;
        cfg.max_batch_tokens = 8;
        cfg.chunked_prefill = false;
        Scheduler sch(cfg);
        sch.submit(20, 4);
        sch.run();
        check(!sch.finished() && sch.step_count() == 100000,
              "run stops at the cap on a stuck config");
    }

    // shared prefix: two seqs with the same 32-token system prompt
    // share 2 blocks. second seq skips computing the shared tokens.
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        uint64_t a = sch.submit(64, 4, 32, 7);
        uint64_t b = sch.submit(64, 4, 32, 7);
        sch.step();
        const Sequence* sa = find(sch, a);
        const Sequence* sb = find(sch, b);
        check(sa->prefilled == 64 && sb->prefilled == 64,
              "shared prefix both seqs fully prefilled on step 0");
        check(sa->first_token_step == 0 && sb->first_token_step == 0,
              "shared prefix both seqs emit first token on step 0");
        check(sch.used_blocks() == 6,
              "shared prefix uses 6 blocks, not 8");
        check(sa->shared_blocks == 2 && sb->shared_blocks == 2,
              "shared prefix each seq holds 2 shared blocks");
        check(sch.metrics().prefix_hits == 1, "shared prefix one hit");
        check(sch.metrics().prefix_tokens_saved == 32,
              "shared prefix saves 32 prefill tokens");
        check(sch.metrics().prefill_tokens == 96,
              "shared prefix only computes 96 of 128 prompt tokens");
        sch.run();
        check(sch.used_blocks() == 0, "shared blocks freed after finish");
    }

    // prefix that does not fill a block: only whole blocks are shared
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(64, 4, 40, 7);
        sch.submit(64, 4, 40, 7);
        sch.step();
        check(sch.metrics().prefix_tokens_saved == 32,
              "partial block prefix shares only 32 of 40 tokens");
        check(sch.metrics().prefill_tokens == 96,
              "partial block prefix computes the 8-token tail twice");
    }

    // different prefix ids never share, even with the same length
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(64, 4, 32, 7);
        sch.submit(64, 4, 32, 8);
        sch.step();
        check(sch.metrics().prefix_hits == 0, "different ids do not share");
        check(sch.used_blocks() == 8, "different ids allocate 8 blocks");
    }

    // same id but different lengths are different prefixes: each
    // registers its own entry, neither attaches to the other
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(64, 4, 32, 7);
        sch.submit(64, 4, 48, 7);
        sch.step();
        check(sch.metrics().prefix_hits == 0,
              "same id different length does not share");
        check(sch.used_blocks() == 8,
              "same id different length allocates 8 blocks");
    }

    // prefix smaller than one block does not share at all
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(32, 4, 8, 7);
        sch.submit(32, 4, 8, 7);
        sch.step();
        check(sch.metrics().prefix_hits == 0, "sub-block prefix no share");
        check(sch.metrics().prefill_tokens == 64,
              "sub-block prefix computes both prompts fully");
    }

    // whole prompt shared: second seq gets its first token for free
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        sch.submit(32, 4, 32, 9);
        uint64_t b = sch.submit(32, 4, 32, 9);
        sch.step();
        check(find(sch, b)->first_token_step == 0,
              "fully shared prompt emits first token on step 0");
        check(sch.metrics().prefill_tokens == 32,
              "fully shared prompt computes 32 tokens once");
        check(sch.used_blocks() == 2, "fully shared prompt uses 2 blocks");
    }

    // prefix entry survives while one holder is still running: the
    // first seq finishes early, the second keeps the shared blocks
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        Scheduler sch(cfg);
        uint64_t a = sch.submit(64, 2, 32, 7);
        uint64_t b = sch.submit(64, 8, 32, 7);
        sch.step();
        sch.step();
        check(find(sch, a)->done, "short seq finishes on step 1");
        check(sch.used_blocks() == 5,
              "shared blocks stay live for the running seq");
        check(!find(sch, b)->done, "long seq still running");
        sch.run();
        check(sch.used_blocks() == 0, "blocks freed when last holder ends");
    }

    // preemption with a shared prefix: the victim drops its hold,
    // recomputes on re-admission, re-attaches, and still finishes
    {
        Config cfg;
        cfg.max_batch_tokens = 512;
        cfg.max_blocks = 6;
        cfg.block_size = 16;
        Scheduler sch(cfg);
        sch.submit(64, 32, 32, 7);
        sch.submit(64, 32, 32, 7);
        sch.run();
        check(sch.finished(), "shared prefix workload finishes under pressure");
        check(sch.metrics().preemptions >= 1,
              "shared prefix workload preempts at least once");
        check(sch.metrics().prefix_hits >= 2,
              "preempted seq re-attaches to the cached prefix");
        check(sch.used_blocks() == 0, "blocks freed after preemption run");
    }

    printf("%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
