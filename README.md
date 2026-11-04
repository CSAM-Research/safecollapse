# safecollapse

Experiment library for the study: **does recursive
training on synthetic data (model collapse) erode the safety alignment of an instruct model?**
It covers the whole study: data preparation, the pilot and inclusion gate, the replace /
subsample / accumulate / steps-matched arms, training (Kazdan recipe), synthetic generation,
safety evaluation with family-separated judges, collapse metrics, and the planned analysis.

> **Safety note.** Runs produce harmful completions. Keep `runs/` private; never publish
> collapsed weights or raw completions.

## Install

```bash
pip install -e ".[gpu,vllm,analysis,capability]"   # GPU machine (Colab, RunPod)
pip install -e ".[analysis]"            # laptop: judging, status, analysis only
```

torch is not a dependency on purpose: Colab and RunPod images ship a CUDA build. Optional
extras: `capability` (lm-evaluation-harness), `diversity` (sentence-transformers), `dev`
(pytest, ruff).

Secrets come from environment variables and are never written to disk: `HF_TOKEN` (gated models
and SORRY-Bench) and `OPENROUTER_API_KEY` (judges). In Colab, add them under the key icon; the
notebook copies them into the environment.

**Before the first run:** accept the terms on Hugging Face for `google/gemma-2-2b-it` and the
SORRY-Bench dataset (`sorry-bench/sorry-bench-202503`), and put about $20 on the OpenRouter key.
Llama-2-7B-Chat is in the registry as a fallback only and is not piloted.

## Use

Step-by-step for RunPod (pods, network volume, secrets): [RUNPOD.md](RUNPOD.md).

```bash
safecollapse prepare      -c pilot          # download + build data, record SHA-256 hashes
safecollapse check-judges -c pilot          # judges live at their pinned providers?
safecollapse pilot        -c pilot          # gate + timings -> runs/pilot/PILOT_REPORT.md
safecollapse run          -c smoke          # PLAN 3.8: Gen 1 -> R2, S2 on 1,000-row rounds
safecollapse run          -c core --gpus 0,1,2,3   # main run, one worker per GPU
safecollapse status       -c core           # what is done / pending (C = checkpoint on disk)
safecollapse judge        -c core           # judge anything not yet judged (CPU, anywhere)
safecollapse analyze      -c core           # tables + planned tests -> runs/main/results/
```

Any config value can be overridden: `--set models=[llama2_7b_chat] --set paths.runs_dir=/workspace/runs`.
Every command is **resumable**: kill it at any point and run it again.

| Config | What it runs (PLAN.md step) |
|---|---|
| `pilot` | Candidates untouched on HarmBench validation; gate; timings (Phase 1) |
| `smoke` | One seed, 1,000-row rounds, Gen 1 -> R2 and S2, end to end (3.8) |
| `core` | Gemma-2-2B-IT: Gen 0, Gen 1, R2-R5, S2-S5, 3 seeds (Phase 4) |
| `optional` | Accumulate A2-A5 and steps-matched O5; reuses core's Gen 0/Gen 1 (6.1) |
| `extend10` | Core arms to generation 10; continues from the kept Gen 5 checkpoints (6.2) |
| `replication` | Core design on Qwen2.5-3B-Instruct (relaxed gate), after the main study (Phase 5) |

Configs live in `src/safecollapse/configs/` (`base.yaml` holds every default with its DESIGN.md
section). Your own YAML can `extends: [core]` and change only what differs. Unknown keys are
errors, and the config refuses models judged by their own family.

## Results in this repository

`results/` holds the numbers behind the paper *Does Model Collapse Erode Safety? Harmful Compliance
Under Recursive Training*:

- `results/<study>/`: `main` (Gemma-2-2B-IT), `base_model` (Gemma-2-2B, Kazdan et al.'s published
  checkpoints) and `replication2` (Qwen2.5-3B-Instruct). Each has the analysis outputs (`summary.csv`,
  `confirmatory_*.json`, `tracking_*`, `table_*.tex`), per-checkpoint `metrics/`, and
  `scores_long.csv.gz`, with one row per prompt, checkpoint and judge, including Gens 6-10.
- `results/judge_validation/`: judge agreement with human labels.
- `results/figures/`: the paper's figures. `scripts/` regenerates its results tables and figures, e.g.
  `python scripts/make_late_table.py results late.tex`.

**Not released:** model weights, model answers, judge replies, benchmark prompts and jailbreak
template text. The study's answers include harmful content, so scores are published as numbers only.
Prompts and templates come from their original public sources, pinned below.

## Public sources

Everything below is fetched from its original source at the exact version the study used:

| Source | Pinned version | Used for | How it is obtained |
|---|---|---|---|
| [HarmBench](https://github.com/centerforaisafety/HarmBench/tree/8e1604d1171fe8a48d8febecd22f600e462bdcdd) | commit `8e1604d` | harmful prompts | `safecollapse prepare` |
| [StrongREJECT](https://github.com/dsbowen/strong_reject/blob/7a551d5b440ec7b75d4f6f5bb7c1719965b76b47/strong_reject/jailbreak_files/wrapping_jailbreaks.json) | commit `7a551d5` | jailbreak templates | automatic on first use; checked against a SHA-256 hash and cached in `~/.cache/safecollapse` |
| [SORRY-Bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/tree/612a4e1f45db8adf884fa62318ddf9fa1c6e75e9) | revision `612a4e1` | harmful prompts | `safecollapse prepare` (accept the dataset's terms first) |
| [XSTest](https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d) | commit `d7bb5bd` | safe prompts | `safecollapse prepare` |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a) | revision `990b271` | training and held-out data | `safecollapse prepare` |

**Joining scores to prompts.** Benchmark IDs can describe the request, so `scores_long.csv.gz` uses
`prompt_key`, the first 12 hex characters of the SHA-256 of the benchmark's prompt ID. After
`safecollapse prepare`, `python scripts/map_prompt_keys.py <data_dir> prompt_keys.csv` writes the
key-to-ID table.

## Checkpoint labels: code vs paper

The code, file paths and data use short labels; the paper names checkpoints by generation, as
Shumailov et al. do ("Generation 0, 1, 2"), with the arm as a superscript from Gen 2 on. Same checkpoints, different
names. The paper's tables are converted on copy by `scripts/relabel_tables.py` (run by `make results`); the code never changes its labels.

| Code / data label | Paper | Meaning |
|---|---|---|
| `G0` (`<model>/gen0`) | Gen 0 | untouched model |
| `G1` (`seed<k>/shared/gen1`) | Gen 1 | first fine-tune on the original data, shared by every arm |
| `R<t>` (`seed<k>/replace/gen<t>`) | Gen t^rep | replace arm, generation t (the collapse condition) |
| `S<t>` (`seed<k>/subsample/gen<t>`) | Gen t^sub | subsample arm, generation t (same data volume, keeps original data) |
| `A<t>` (`seed<k>/accumulate/gen<t>`) | Gen t^acc | accumulate arm, generation t (base category; grows with t) |

Hypotheses in paper wording: H1 Gen 5^rep > Gen 1; H2 a positive slope over Gen 1 and replace
Gens 2-5; H3 Gen 5^sub < Gen 5^rep (base: Gen 5^acc); RQ4 over-refusal at Gen 5^rep
vs Gen 1.
"S" is always an arm, never a seed.

## Architecture

```
configs (YAML) ──> config.Config (frozen, validated)
                         │
plan.build_plan ──> [Node(model, seed, arm, gen)]  (DAG: sources() = whose answers a node trains on)
                         │
pipeline.Runner ──> per node, in order, each stage skipped if its marker exists:
   data      arms.training_set ............................ CPU, pure
   train     training.train (fresh fine-tune of Gen 0) .... GPU subprocess
   gpu       generation.run_jobs (eval sets + synthetic) .. GPU subprocess, one model load
             metrics.collapse (held-out NLL, tail loss)
   judge     judges.runner -> OpenRouter, cached .......... background thread
   cleanup   only if keep_checkpoints: false (default keeps every checkpoint)
then classify (HarmBench classifier, all nodes, one load) and analyze (analysis.*)
```

**Parallel runs.** `--gpus 0,1,2,3` starts one worker per GPU. Workers share the study folder
and take whichever checkpoint is ready (the synthetic data it trains on exists) and not locked by
another worker. After Gen 1, every (seed, arm) chain is independent, so the core run of one
model has 6 chains and two models have 12: on 4 GPUs the wall-clock time drops about 4x for the
same GPU-hours. Within a worker, API judging runs in a background thread (16 calls at a time).
Also works across machines that share a volume (`run --break-locks` clears locks left by a
machine that died).

Design choices, briefly:

- **One process per GPU stage.** vLLM does not release GPU memory reliably; a subprocess does.
  It also lets training run under `accelerate launch` (`training.launcher`) for 7B models.
- **Markers after outputs, atomic writes.** A stage is done only when `.done/<stage>` exists,
  written after its files, so a crash redoes at most one stage.
- **The recursion lives in data.** Every checkpoint fine-tunes Gen 0; `arms.py` is a few pure
  functions that decide what each checkpoint trains on, and they are unit-tested.
- **Same tokens in training and generation.** `modeling.prompt_ids` renders the native chat
  template once; training masks exactly those tokens (loss on the answer only).
- **Judge calls are never paid twice.** A disk cache keyed by (task, judge, provider pin, judge
  prompt); providers pinned with fallbacks off; the provider that served each call is recorded.
- **One protocol per study folder.** `runs/<name>/protocol.json` stores the settings that define
  a checkpoint; a run with different settings refuses to write into the same folder.

Run folder layout:

```
runs/<name>/<model>/gen0/                        Gen 0 (untouched; no seed)
runs/<name>/<model>/seed<k>/shared/gen1/         Gen 1, shared by every arm
runs/<name>/<model>/seed<k>/<arm>/gen<t>/
    train.jsonl  synthetic.jsonl  checkpoint/  manifest.json
    eval/<benchmark>/completions.jsonl, <task>__<judge>.jsonl
    metrics/*.json  .done/
runs/<name>/results/                             analyze output
runs/_judge_cache/<judge>.jsonl                  shared by all studies
```

Each `manifest.json` records per stage: library versions, GPU, model revision, data and output
hashes, timings, judge model, provider pin and the providers that served the calls.

## Decisions taken in the code (check with the design)

- **Round size.** HelpSteer2 train has 20,324 rows but only 10,161 unique prompts, so a
  12,500-row round samples rows, not prompts (some prompts appear with both responses).
  `data.one_response_per_prompt: true` gives one response per prompt, capped at 10,161 rows.
- **Multi-turn prompts.** About 29% of HelpSteer2 prompts are multi-turn (NeMo format); they are
  parsed into chat turns for the native template.
- **SORRY-Bench fulfilment** uses SORRY-Bench's official judge (the authors' fine-tuned
  Mistral-7B, local, gated: accept its terms on Hugging Face), with their prompt and decoding.
- **XSTest over-refusal** uses the XSTest authors' own classifier prompt (three classes) with
  the primary judge model; both refusal classes count.
- **Tail loss threshold** = 95th percentile of the original data's per-example perplexity under
  Gen 0, per seed.
- **Empty synthetic answers** are kept: the model then trains to end the turn immediately, which
  is part of what collapse feeds back.
- **Capability benchmark:** IFEval (proposed; open decision 5), via lm-evaluation-harness.
- **Mixed precision.** Training keeps float32 master weights and computes in bfloat16; pure
  bfloat16 weights would silently drop most updates at learning rate 8e-6. Needs an 80 GB A100
  for Gemma-2-2B. Checkpoints are saved in bfloat16.
- **Checkpoints are all kept** (`keep_checkpoints: true`): budget about 5 GB per Gemma-2-2B
  checkpoint (about 140 GB for the core run), 13.5 GB per Llama-2-7B checkpoint.

## Tests

```bash
pip install -e ".[analysis,dev]" && pytest -q && ruff check src tests
```

The tests cover pure logic only (configs, the plan, arms, judge parsing and caching, metrics,
statistics, chat masking with a stand-in tokenizer). Nothing in this library runs a model on a
CPU: GPU stages stop if no CUDA GPU with bfloat16 is present.
