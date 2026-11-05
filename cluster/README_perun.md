# Setting up the comparison package on Perun

Run the steps in order. Each step ends with a check; do not continue past a
failed check. `local$` = Git Bash on the Windows machine in `Q:\new_try`;
`perun$` = login node.

## 1. Disk space (Qwen3-VL alone is ~60 GB, features ~20 GB)

```bash
perun$ df -h /mnt/project/perun2601420 /mnt/home/rupyke723
perun$ du -sh /mnt/home/rupyke723/.cache/huggingface
```

Check: at least ~120 GB free where HF_HOME lives. If home is small, move
HF_HOME to the project volume and change it in every `cluster/*` script.

## 2. Clone the repository

```bash
perun$ cd /mnt/project/perun2601420
perun$ git clone <github-url> comparison_study
perun$ cd comparison_study && mkdir -p logs
perun$ ls /mnt/project/perun2601420/data/video/     # fix the CHECK paths in config.perun.yaml
```

Every script on Perun runs with `--config config.perun.yaml`. Fix the paths
there once and commit them; `git pull` then brings new code without touching
any cluster state. `folds.csv` comes with the repository. `videos.csv` is
regenerated on Perun after step 3 (`s01_build_videos.py`), and its
`videos_stats.json` must come out identical to the committed one.

The repository is private, so cloning on Perun needs credentials: a GitHub
personal access token (fine-grained, read-only on this repository) or an SSH
key added to GitHub.

## 3. Upload the raw videos (11 GB, 3198 mp4; needed for audio and Qwen3-VL)

```bash
local$ cd "/q/new_try/EXIST 2026 Videos Dataset"
local$ tar cf - training/videos test/videos | \
        ssh rupyke723@login01.perun.tuke.sk "mkdir -p /mnt/project/perun2601420/data/videos && tar xf - -C /mnt/project/perun2601420/data/videos"
perun$ ls /mnt/project/perun2601420/data/videos/training/videos | wc -l    # expect 2524
perun$ ls /mnt/project/perun2601420/data/videos/test/videos | wc -l        # expect 674
```

If the layout ends up different, set `train_videos_dir` / `test_videos_dir`
in `config.yaml` to wherever the mp4 files are; `check_env` counts them.

## 4. Packages in the existing training env (`exist_vt`)

```bash
perun$ export PATH="/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin:$PATH"
perun$ python -c "import torch; print(torch.__version__, torch.version.cuda)"
perun$ python -m pip install PyEvALL==0.1.74 imagehash av pyyaml faiss-cpu lightgbm \
        sentence-transformers torch_geometric
```

Always `python -m pip`, never `pip`: the env's console scripts (`pip`,
`huggingface-cli`, `conda` itself) have shebangs pointing at the dead
`/mnt/data/home` prefix and fail with "bad interpreter".

`torch_geometric` needs no compiled extensions for GATv2Conv / HypergraphConv,
so the plain pip package works with the installed torch.

## 5. Two separate environments (they pin their own torch; keep them apart)

`conda create` is broken for the same reason, so both are plain venvs built
from exist_vt's Python 3.11 and kept on the project volume:

```bash
perun$ VT=/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python
perun$ mkdir -p /mnt/project/perun2601420/envs
perun$ $VT -m venv /mnt/project/perun2601420/envs/exist_asr
perun$ /mnt/project/perun2601420/envs/exist_asr/bin/python -m pip install -U pip
perun$ /mnt/project/perun2601420/envs/exist_asr/bin/python -m pip install whisperx silero-vad faster-whisper av pyyaml pandas h5py
perun$ $VT -m venv /mnt/project/perun2601420/envs/exist_vllm011
perun$ /mnt/project/perun2601420/envs/exist_vllm011/bin/python -m pip install -U pip
perun$ /mnt/project/perun2601420/envs/exist_vllm011/bin/python -m pip install "vllm==0.11.0" \
        "transformers>=4.57,<5" qwen-vl-utils pyyaml pandas h5py opencv-python-headless scenedetect imagehash
```

Environments are always used by absolute interpreter path, as above and in
the job scripts.

What did not work, so it is not repeated:

- **vLLM 0.32** installed without its compiled kernels (`No module named
  'vllm._C'`) and tried to build them at start-up, which fails on nodes
  without `nvcc`. 0.11.0 ships the kernels for CUDA 12.8 and supports
  Qwen3-VL. `transformers` stays below 5, which 0.11 predates.
- **libstdc++:** the venvs use exist_vt's conda Python, whose `sqlite3`
  (imported by vLLM and others) links an ICU that needs a newer libstdc++
  than `/lib64` has. Every job script using a venv sets
  `LD_PRELOAD=/mnt/home/rupyke723/miniconda3/envs/exist_vt/lib/libstdc++.so.6`.
  Do the same when testing by hand.
- **No `libcuda` on the login node:** importing vLLM's kernels there fails.
  Test GPU imports in a job.

## 6. Models (login node, has internet)

```bash
perun$ bash cluster/download_models.sh text
perun$ bash cluster/download_models.sh vision
perun$ bash cluster/download_models.sh audio
perun$ bash cluster/download_models.sh qwen       # long; can run in tmux/screen
```

`audio` also fetches what compute nodes cannot download themselves: the
torchaudio aligners WhisperX uses for English and Spanish (into the torch hub
cache) and NLTK `punkt` (into `~/nltk_data`).

## 7. Check everything (1 job, ~10 min)

```bash
perun$ sbatch cluster/check_env.slurm
perun$ tail -f logs/cmp_check_env_*.out
```

Check: every env says `ALL OK`, the GPU is listed, and the log ends with
`SELFTEST PASSED`. The selftest compares our fast ICM with PyEvALL on this
machine's Python, which is the one that matters.

## 8. Encoder features for the graphs (Step 6)

Frozen XLM-R-large, VideoMAE (biases fixed), Whisper large-v3 encoder, BGE-M3
and SigLIP 2, one array task each, into `outputs/features/<stage>.h5`. Needs
`outputs/moments/moments.jsonl` and the v3 JSONs; models from step 6 (`text`,
`vision`, `audio`).

```bash
perun$ SMOKE=1 sbatch --time=0:30:00 cluster/features.slurm   # 8 videos -> outputs/features_smoke
perun$ python src/s06_features.py --config config.perun.yaml --stage check --limit 8 --out-dir outputs/features_smoke
perun$ sbatch cluster/features.slurm                          # all 3198 videos
perun$ python src/s06_features.py --config config.perun.yaml --stage check
```

`check` needs no GPU: every file holds every video, shapes match, no NaN,
speech masks agree with the moments. A failed or timed-out stage resumes
where it stopped when re-submitted (`--array=<i>` for just that stage).

Then the candidate neighbours for links between videos (Step 7; CPU, under a
minute, fine on the login node). It needs `bge.h5` and `siglip.h5`:

```bash
perun$ python src/s07_candidates.py --config config.perun.yaml   # -> outputs/candidates.h5
```

## 9. Audio-visual models for W3's links (smoke, `src/av_edges.py`)

Two models that watch the whole video with its sound: Qwen3-Omni-30B-A3B
(transformers, in the vLLM env) and video-SALMONN 2+ 7B (the authors' code,
its own env: transformers 4.51 and torch 2.7). Login node:

```bash
perun$ /mnt/project/perun2601420/envs/exist_vllm011/bin/python -m pip install av
perun$ VT=/mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python
perun$ $VT -m venv /mnt/project/perun2601420/envs/exist_salmonn
perun$ /mnt/project/perun2601420/envs/exist_salmonn/bin/python -m pip install -U pip
perun$ /mnt/project/perun2601420/envs/exist_salmonn/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
        torchaudio==2.7.1 transformers==4.51.3 tokenizers==0.21.0 accelerate==1.7.0 peft==0.15.2 \
        liger_kernel==0.5.10 decord==0.6.0 numpy==1.24.4 av pyyaml pandas h5py pillow
perun$ bash cluster/download_models.sh av      # clones the SALMONN code, fetches both models (~75 GB)
```

No flash-attn (both run with `sdpa`) and no torchcodec: `av_edges.py` decodes
SALMONN's audio with PyAV and lets its video go through decord. Then:

```bash
perun$ MODEL=omni    sbatch cluster/av_edges.slurm
perun$ MODEL=salmonn sbatch cluster/av_edges.slurm
```

Each writes `outputs/w3/edges_smoke_av_<model>.jsonl`, prints W3's summary
and the agreement with `outputs/w3/edges.jsonl`, and a review page
`edges_smoke_av_<model>_vs_edges_review.html`.

## 10. Robustness check: a plainer message function (choices.md)

C1, C3 and W2 again, with GATv2 replaced by a plain mean of the neighbours'
messages (`CONV=mean`) or the same mean weighted by the edge's cosine
(`CONV=wmean`); same 20 trials and 5 seeds, runs in `runs/graphs_v2_<conv>`.
From the package root, one queue per variant (they share Perun's 4 job slots):

```bash
perun$ CONV=wmean DRY=1 bash cluster/queue_graphs.sh       # what it would submit
perun$ CONV=wmean nohup bash cluster/queue_graphs.sh > logs/queue_graphs_wmean.log 2>&1 &
perun$ CONV=mean  nohup bash cluster/queue_graphs.sh > logs/queue_graphs_mean.log  2>&1 &
```

When a queue prints `everything finished`:

```bash
perun$ nohup /mnt/home/rupyke723/miniconda3/envs/exist_vt/bin/python src/graph_report.py --config config.perun.yaml --runs runs/graphs_v2_wmean \
        --out results/graphs_v2_wmean > results/graph_report_wmean.txt 2>&1 &
```
