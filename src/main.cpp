#include "../src/sched.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace batchd;

static const char* VERSION = "1.1.0";

static void usage() {
    printf("batchd %s - continuous batching for llm inference\n", VERSION);
    printf("usage:\n");
    printf("  batchd demo [--max-batch-tokens N] [--max-blocks N]\n");
    printf("  batchd bench [--seqs N] [--prompt N] [--max-tokens N]\n");
    printf("               [--max-batch-tokens N] [--max-blocks N] [--seed N]\n");
    printf("               [--shared-prefix N]\n");
    printf("  batchd verify\n");
    printf("  batchd --version\n");
}

static int flag_val(int argc, char** argv, const char* name, int dflt) {
    for (int i = 0; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    }
    return dflt;
}

static void print_metrics(const Scheduler& sch) {
    Metrics m = sch.metrics();
    printf("steps: %d\n", m.steps);
    printf("prefill tokens: %d, decode tokens: %d\n", m.prefill_tokens,
           m.decode_tokens);
    printf("throughput: %.2f tokens/step\n",
           m.steps ? (double)(m.prefill_tokens + m.decode_tokens) / m.steps : 0);
    printf("mean ttft: %.2f steps, mean tpot: %.2f steps/token\n", m.mean_ttft,
           m.mean_tpot);
    printf("preemptions: %d, peak batch: %d\n", m.preemptions, m.peak_running);
    if (m.prefix_hits)
        printf("prefix cache: %d hits, %d prompt tokens skipped\n",
               m.prefix_hits, m.prefix_tokens_saved);
}

static int demo(int argc, char** argv) {
    Config cfg;
    cfg.max_batch_tokens = flag_val(argc, argv, "--max-batch-tokens", 64);
    cfg.max_blocks = flag_val(argc, argv, "--max-blocks", 32);
    cfg.max_seqs = 4;
    cfg.block_size = 16;
    Scheduler sch(cfg);
    int prompts[] = {64, 96, 128, 8, 64, 200};
    int maxtok[] = {8, 12, 10, 24, 16, 6};
    // the first three share a 48-token system prompt, #5 and #6 share
    // a 32-token one, #4 shares nothing. shared blocks allocate once.
    int prefix[] = {48, 48, 48, 0, 32, 32};
    uint64_t pid[] = {1, 1, 1, 0, 2, 2};
    for (int i = 0; i < 6; i++)
        sch.submit(prompts[i], maxtok[i], prefix[i], pid[i]);
    printf("step  running (prefilled/prompt, generated)  blocks\n");
    while (!sch.finished()) {
        sch.step();
        printf("%4d  ", sch.step_count() - 1);
        for (const auto& s : sch.sequences()) {
            if (s.done) continue;
            printf("#%llu %d/%d,%d", (unsigned long long)s.id, s.prefilled,
                   s.prompt_len, s.generated);
            if (s.shared_blocks) printf("+%dsh", s.shared_blocks);
            printf("  ");
        }
        printf(" %d/%d\n", sch.used_blocks(), cfg.max_blocks);
    }
    printf("\n");
    print_metrics(sch);
    return 0;
}

static int bench(int argc, char** argv) {
    int nseqs = flag_val(argc, argv, "--seqs", 64);
    int prompt = flag_val(argc, argv, "--prompt", 128);
    int seed = flag_val(argc, argv, "--seed", 7);
    int shared = flag_val(argc, argv, "--shared-prefix", 0);
    Config cfg;
    cfg.max_batch_tokens = flag_val(argc, argv, "--max-batch-tokens", 512);
    cfg.max_blocks = flag_val(argc, argv, "--max-blocks", 1024);
    cfg.default_max_tokens = flag_val(argc, argv, "--max-tokens", 32);
    cfg.max_seqs = 32;
    cfg.block_size = 16;
    Scheduler sch(cfg);
    // prompt lengths jitter around the base so the batch is uneven,
    // like real traffic. deterministic per seed.
    unsigned state = (unsigned)seed * 2654435761u + 1;
    for (int i = 0; i < nseqs; i++) {
        state = state * 1664525u + 1013904223u;
        int len = prompt / 2 + (int)(state % (unsigned)prompt);
        if (len < 1) len = 1;
        // --shared-prefix N: groups of four share the first N tokens
        // of their prompt, like a common system prompt.
        uint64_t pid = shared > 0 ? (uint64_t)(i / 4 + 1) : 0;
        sch.submit(len, -1, shared, pid);
    }
    auto t0 = std::chrono::steady_clock::now();
    sch.run();
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    Metrics m = sch.metrics();
    printf("seqs: %d, prompt base: %d, seed: %d\n", nseqs, prompt, seed);
    printf("steps: %d, wall: %.2f ms\n", m.steps, ms);
    print_metrics(sch);
    return sch.finished() ? 0 : 1;
}

static int verify() {
    // fixed workload, check the invariants hold after every step.
    Config cfg;
    cfg.max_batch_tokens = 64;
    cfg.max_seqs = 4;
    cfg.block_size = 16;
    cfg.max_blocks = 32;
    Scheduler sch(cfg);
    int prompts[] = {8, 40, 128, 16, 64, 200, 33, 90};
    for (int i = 0; i < 8; i++) sch.submit(prompts[i], 16);
    bool ok = true;
    while (!sch.finished()) {
        sch.step();
        if (sch.used_blocks() > cfg.max_blocks) ok = false;
        if (sch.running_count() > cfg.max_seqs) ok = false;
        if (sch.last_step_tokens() > cfg.max_batch_tokens) ok = false;
        for (const auto& s : sch.sequences()) {
            if (s.generated > s.max_tokens) ok = false;
            if (s.prefilled > s.prompt_len) ok = false;
        }
    }
    for (const auto& s : sch.sequences()) {
        if (!s.done || s.generated < 1) ok = false;
    }
    // shared prefix workload: two seqs with the same 32-token start
    // should share blocks and skip computing the prefix twice.
    {
        Config pcfg;
        pcfg.max_batch_tokens = 512;
        Scheduler psch(pcfg);
        psch.submit(64, 4, 32, 7);
        psch.submit(64, 4, 32, 7);
        psch.run();
        if (!psch.finished()) ok = false;
        if (psch.metrics().prefix_hits != 1) ok = false;
        if (psch.metrics().prefix_tokens_saved != 32) ok = false;
        if (psch.metrics().prefill_tokens != 96) ok = false;
        if (psch.used_blocks() != 0) ok = false;
    }
    printf("verify: %s (%d steps, %d preemptions)\n", ok ? "ok" : "FAILED",
           sch.step_count(), sch.metrics().preemptions);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    if (std::strcmp(argv[1], "--version") == 0) {
        printf("batchd %s\n", VERSION);
        return 0;
    }
    if (std::strcmp(argv[1], "demo") == 0) return demo(argc, argv);
    if (std::strcmp(argv[1], "bench") == 0) return bench(argc, argv);
    if (std::strcmp(argv[1], "verify") == 0) return verify();
    usage();
    return 1;
}
