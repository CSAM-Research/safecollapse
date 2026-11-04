# Running on RunPod

Use **Pods** (GPU Cloud), not Serverless. Serverless is for short inference requests that scale to
zero; this study runs long training jobs that keep state on disk.

Everything lives on one **network volume** mounted at `/workspace`: code, Python environment,
model cache, data, checkpoints and results. Pods are disposable: terminate them when idle; the
volume keeps everything.

## Once

1. **Secrets** (Settings -> Secrets): create `hf_token` (your Hugging Face read token) and
   `openrouter_api_key` (your OpenRouter key).
2. **Network volume** (Storage -> Network Volumes -> New): about **300 GB**, in a data center that
   lists **A100 SXM 80GB** (or A100 PCIe 80GB) as available, ideally several of them. Pods can
   only attach a volume in the same data center.

## Each pod

Pods -> Deploy:
- **Network volume:** select yours (this filters to its data center).
- **GPU:** A100 80GB. 1 GPU for the pilot and smoke test; **4 GPUs** in one pod for the main run.
- **Template:** Runpod PyTorch 2.8.0 (any recent PyTorch/CUDA 12 template works; vLLM brings its own torch).
- **Pricing:** On-Demand, not Spot (Spot pods can be taken away mid-run).
- **Container disk:** 30 GB is enough (everything big goes to `/workspace`).
- **Environment variables** (Edit Template -> Environment Variables):

  | Key | Value |
  |---|---|
  | `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
  | `OPENROUTER_API_KEY` | `{{ RUNPOD_SECRET_openrouter_api_key }}` |
  | `HF_HOME` | `/workspace/hf_cache` (models download once and persist) |

Connect -> Jupyter Lab (or Web Terminal). Upload `safecollapse.zip` to `/workspace` (drag it into
Jupyter's file browser), then open a terminal.

**First pod only: install into the volume (about 10 minutes)**

```bash
cd /workspace && unzip -o safecollapse.zip
(pip install -q uv || pip install -q --break-system-packages uv)
uv venv /workspace/venv --python 3.12 && source /workspace/venv/bin/activate
uv pip install vllm && uv pip install -e "./safecollapse[gpu,analysis,capability]"
safecollapse --help    # should list prepare, check-judges, pilot, run, ...
```

**Every pod: activate and work from `/workspace`** (results go to `/workspace/runs` and
`/workspace/data`)

```bash
source /workspace/venv/bin/activate && cd /workspace
apt-get update -qq && apt-get install -y -qq tmux   # runs survive closing the browser
tmux new -s run                                      # reattach later: tmux attach -t run
```

## Steps

| # | Pod | Command | Time |
|---|---|---|---|
| 1 | 1x A100 | `safecollapse prepare -c pilot` | 2 min |
| 2 | | `safecollapse check-judges -c pilot` (all lines must say `live=True verdict_ok=True`) | 1 min |
| 3 | | `safecollapse pilot -c pilot` -> `runs/pilot/PILOT_REPORT.md` | ~1 h |
| 4 | | `safecollapse run -c smoke`; press Ctrl-C once midway, rerun, check `safecollapse status -c smoke` | ~1-2 h |
| | | **Terminate the pod.** | |
| 5 | 4x A100 | `safecollapse run -c core --gpus 0,1,2,3` | ~13-16 h |
| 6 | any | `safecollapse status -c core` (L = running, C = checkpoint on disk) | |
| 7 | 1x A100 or CPU pod | `safecollapse analyze -c core` -> `runs/main/results/` | minutes |

If anything stops (pod, worker, network), run the same command again: it resumes.

If no 4-GPU pod is available, start up to four 1-GPU pods on the same volume and run
`safecollapse run -c core` in each; they share the work through lock files. If a pod died while
holding a lock, run once with `--break-locks` (only when no other pod is running).

## Money

- Pods bill per second while running, including when idle: **terminate when nothing is running**.
- The network volume bills monthly while it exists (about $0.07 per GB per month).
- Keep enough balance: RunPod stops pods when the balance runs out, which would interrupt a run
  (it resumes, but the time is lost).
- Download `runs/main/results/` and the reports to your laptop. Never download or share
  completions or checkpoints outside the volume.
