# Graph construction comparison (thesis part 2)

Compares eleven ways of building the graph for multimodal sexism detection in
EXIST 2026 TikTok videos, with everything except graph construction held
fixed. The full plan is `PLAN.md`; what the data
actually looks like, and the problems found in it, is in `data_note.md`.

## Layout

| path | what |
|---|---|
| `src/s01_build_videos.py` | Step 1: one row per video → `outputs/videos.csv` |
| `src/s02_near_duplicates.py` | near-duplicate groups (pHash over H5 clips) |
| `src/s03_build_folds.py` | Step 2: author-grouped outer and inner folds → `outputs/folds.csv` |
| `src/evaluate.py`, `src/icm_fast.py` | Step 3: metrics, cluster bootstrap, confidence-matched controls |
| `src/text_clean.py` | removes leaked LLM instructions and ASR loops from the organisers' `text` |
| `src/s04_asr.py` | Whisper large-v3 + WhisperX: speech with word times, language ID, hallucination filters |
| `src/s05_moments.py` | 4 moments per video (one H5 clip each), timed speech, ASR reliability, VLM transcript |
| `src/vlm_frames.py`, `src/vlm_describe.py`, `src/vlm/` | regenerated VLM descriptions: part 1's sampler and prompt, Qwen3-VL-30B via vLLM |
| `src/asr_bakeoff.py`, `src/asr_review.py`, `src/serve_review.py` | the ASR comparison and the listening review that set the ASR design |
| `src/convert_part1_oof.py` | part-1 OOF predictions → our prediction format |
| `src/check_env.py` | environment check (packages, GPU, paths, models) |
| `cluster/` | Perun setup (`README_perun.md`) and SLURM jobs |
| `config.yaml` / `config.perun.yaml` | local / cluster paths |

## Running locally

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/s01_build_videos.py --config config.yaml
.venv/Scripts/python src/s02_near_duplicates.py --config config.yaml
.venv/Scripts/python src/s03_build_folds.py --config config.yaml --scheme grouped
.venv/Scripts/python src/evaluate.py --config config.yaml selftest
```

## Prediction format

Every model writes one CSV per task: `id_EXIST`, `p_<CLASS>` for each class,
optional `hard`. `evaluate.py score | compare | controls` consume it.

On Perun, follow `cluster/README_perun.md` and pass `--config config.perun.yaml`.
