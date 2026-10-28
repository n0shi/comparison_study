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
