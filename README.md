# RSI Flow Lab

A compact testbed for recursive self-improvement experiments using 2D flow matching, automated fit checks, and particle trajectory visualizations.

**Status:** training and evaluation baseline established. An automated recursive self-improvement loop is planned; it is not implemented yet.

## Experiments

| ID | Method / change | Dataset | Validation MSE ↓ | Mean SW1 ↓ | Classes passing ↑ | Finding |
|---|---|---|---:|---:|---:|---|
| [001](#exp-001--flow-matching-baseline) | Conditional MLP flow matching; baseline | `checkerboard_v2`, 100 classes | 1.0902 | 0.0340 | 21/100 | Global distributions improve, but most classes fail at least one fit check. |

### EXP-001 — Flow matching baseline

**Question.** Can a small conditional MLP fit 100 noisy 2D shape distributions?

**Method.** Generate target coordinates online from ten shape families: checkerboards, ellipses, spirals, roses, polygons, stars, Lissajous curves, waves, Gaussian rings, and superellipses. Each family has ten parameter variants. Add Gaussian coordinate noise with standard deviation 0.035.

Train on `x_t = (1 − t)x_0 + tx_1`, where `x_0 ~ N(0, I)` and `x_1` is a target point. The MLP takes `(x_t, t, label)` and predicts velocity, supervised by `MSE(v, x_1 − x_0)`. Generate point clouds by integrating the learned velocity with Heun's method.

| Setting | Value |
|---|---|
| Model | 4 hidden layers × 256 units, SiLU, class embeddings, spatial/time Fourier features |
| Training | 15,000 steps; batch 2,048; AdamW; LR 0.001; 500-step warmup + cosine decay |
| Sampling | 100 Heun steps; 2,000 points per class |
| Run | Seed 42; PyTorch 2.8.0; Apple MPS; 2026-09-30 |
| Duration | 425.5 s, including training progress sampling/plots; excluding final sampling/evaluation |

**Inference.** The same particles move from Gaussian noise to the generated distribution. Colors remain fixed. This animation shows classes 09, 19, …, 99, with 300 particles per class and 101 frames. Weights stay fixed throughout the animation.

![EXP-001: particle trajectories from noise to generated shapes](experiments/001-baseline/inference.gif)

**Result.** Validation MSE fell from **1.6325 to 1.0902**. Mean sliced Wasserstein-1 was **0.0340**, versus **0.2331** for Gaussian noise. Only **21/100 classes passed** the combined fit checks: ellipses 9/10, polygons 1/10, Gaussian rings 2/10, and superellipses 9/10. All other families passed 0/10. This is a working baseline with substantial fitting errors, not a solved benchmark.

[Target vs. generated](experiments/001-baseline/comparison.png) · [Per-class results](experiments/001-baseline/quality.csv) · [Exact configuration](experiments/001-baseline/config.json) · [Metrics](experiments/001-baseline/metrics.json) · [Acceptance thresholds](experiments/001-baseline/quality_report.json)

## Evaluation protocol

Each class must pass **all four checks**: sliced Wasserstein-1 (global distribution), nearest-distance precision (proximity to target support), coverage (missing regions), and multiscale grid JS divergence (local density). Thresholds are calibrated against independent target samples: at most 1,000 points per class, five calibration draws, distance tolerance ×1.5, and probability slack 0.05. These are heuristic acceptance checks, not proof of distributional equivalence.

Keep the dataset version, evaluation settings, sampling budget, and seeds fixed when comparing experiments. Report method changes and compute budgets explicitly. EXP-001 is a single-seed result.

## Reproduce

Use a Python environment with [requirements.txt](requirements.txt) installed. Run from the repository root:

```bash
python run_train.py --device mps --num-classes 100 \
  --hidden-dim 256 --num-layers 4 --batch-size 2048 \
  --steps 15000 --warmup-steps 500 --learning-rate 0.001 --seed 42 \
  --sample-points 2000 --sample-steps 100 \
  --output-dir outputs/exp001_reproduction \
  --save-progress --save-every 1000 --log-every 500 \
  --inference-animation --animation-points 300 --inference-gif-duration-ms 50

python run_evaluate.py --output-dir outputs/exp001_reproduction
```

Use `--device cpu` or `--device cuda` on other hardware. Full checkpoints and generated files stay in the ignored `outputs/` directory. Selected results under `experiments/` are kept for GitHub. `progress.gif` shows training evolution; `inference.gif` shows particle movement through one fixed model.

## Add an experiment

Append one table row and one short entry: **question → method/change → configuration → results → inference GIF → finding**. Save selected artifacts in `experiments/002-<name>/` with the exact config and evaluation report. Retain failures and keep acceptance thresholds unchanged across comparisons.
