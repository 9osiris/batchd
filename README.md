batchd - continuous batching for llm inference [educational project]

disclaimer:
this is a scheduler, not a model server. there is no real model in
here, the "model" is a hash function that returns a deterministic
token, and the steps are logical, not wall clock. i built it to
understand how continuous batching actually works, not to serve
anything. dont use it for real inference.

why i made this:
every model i care about runs more than one sequence at a time, and
the naive way is to run them one after another and waste the whole
batch on padding. the real servers admit sequences as they arrive,
prefill them in chunks against a token budget, then decode one token
per step for everything running at once. the interesting part is the
kv memory: blocks are the currency, a long prompt eats a bunch of
them, and when decode needs one more block and none is free,
something has to get evicted and recomputed. i wrote the scheduler
first, then found the livelock the honest way: my first version let
prefill preempt too, and two long sequences kept evicting each other
forever, neither one ever decoding past its first token. prefill now
waits for a roomier step and only decode preempts, which is how the
real systems avoid it. c++ because the inference lane is c++ now:
tokkit turns text into tokens, flash mixes them, kvblocks caches the
keys and values, batchd is the scheduler that decides which
sequences get to run together. one tradeoff the demo shows on
purpose: a big prefill chunk eats the whole step budget and decode
starves that step, so ttft and tpot fight each other. that tension is
real, it is not a bug.

the guide:
build it:
  make

run the tests:
  make test

watch a small workload step by step:
  ./batchd demo

bench a bigger workload:
  ./batchd bench --seqs 64 --prompt 128
  ./batchd bench --seqs 64 --prompt 128 --max-batch-tokens 128 --max-blocks 64
  ./batchd bench --seqs 64 --prompt 128 --shared-prefix 64

check the invariants on a fixed workload:
  ./batchd verify

flags:
  demo takes --max-batch-tokens and --max-blocks.
  bench takes --seqs, --prompt, --max-tokens, --max-batch-tokens,
  --max-blocks, --seed, --shared-prefix. prompt lengths jitter around
  the base so the batch is uneven, like real traffic, and stay
  deterministic per seed. --shared-prefix N makes groups of four
  share the first N tokens of their prompt, like a common system
  prompt, so bench shows what prefix caching is worth.

how it works:
  a step has a token budget. waiting sequences get admitted while the
  running batch has room, then prefill runs in submit order, chunked
  to the budget, and a sequence that finishes prefill emits its first
  token in the same step. decode then gives one token per running
  sequence that already prefilled, skipping the ones that just got
  their first token. kv blocks are 16 tokens each by default. a
  sequence that needs a block for decode and finds none free preempts
  the youngest other running sequence, which goes back to waiting
  with its progress wiped and recomputes when admitted again.
  metrics are per sequence: ttft is steps from arrival to first
  token, tpot is steps per output token after the first. run() caps
  at 100000 steps so a stuck config (chunked prefill off and the
  budget smaller than the longest prompt, or blocks too small to hold
  one prompt) stops instead of spinning forever.

prefix sharing:
  sequences that start with the same prompt (a system prompt, a
  few-shot header) can share the kv blocks for that start instead of
  each allocating their own. submit takes a prefix length and a
  prefix id, sequences with the same id share the first prefix_len
  tokens. only whole blocks are shared, the tail of a prefix that
  does not fill a block stays private, and the same id with a
  different length is a different prefix. the first sequence to
  prefill past the shared part registers those blocks in a cache,
  later sequences with the same id attach to it, skip computing the
  shared tokens, and hold the blocks by refcount. the entry frees
  when the last sequence lets go, either by finishing or by being
  preempted. a preempted sequence drops its hold and re-attaches
  when it is admitted again, if the entry is still alive. a sequence
  whose whole prompt is shared gets its first token without any
  prefill compute. bench --shared-prefix 64 on 64 seqs skips 3072
  prompt tokens (4832 computed instead of 7904) with 48 cache hits.

follow-ups:
  decode-priority scheduling so a big prefill cant starve decode,
  priority lanes for short prompts.
