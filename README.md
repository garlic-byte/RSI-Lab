# RSI Flow Lab

A compact testbed for recursive self-improvement experiments using 2D flow matching, automated fit checks, and particle trajectory visualizations.

**Status:** training and evaluation baseline established. An automated recursive self-improvement loop is planned; it is not implemented yet.

## Experiments

| ID | Method / change | Dataset | Validation MSE ↓ | Mean SW1 ↓ | Classes passing ↑ | Finding |
|---|---|---|---:|---:|---:|---|
| [001](#exp-001--flow-matching-baseline) | Conditional MLP flow matching; baseline | `checkerboard_v2`, 100 classes | 1.0902 | 0.0340 | 21/100 | Global distributions improve, but most classes fail at least one fit check. |
| [002](#exp-002--transformer-backbone) | Three-token Transformer; same training settings | `checkerboard_v2`, 100 classes | 1.1243 | 0.0315 | 36/100 | More classes pass with similar parameter count, at higher runtime cost. |
| [003](#exp-003--cnn-backbone) | Three-position Conv1d; same training settings | `checkerboard_v2`, 100 classes | 1.1062 | 0.0322 | 28/100 | Fit improves over MLP, trails Transformer, with the highest measured runtime. |

### Fit-check pass rate by shape family

Each cell shows **passing classes / 10 variants**. A class passes only when all four fit checks pass; these are class-level acceptance rates, not per-point accuracy.

| Class IDs | Shape family | MLP (001) | Transformer (002) | CNN (003) |
|---|---|---:|---:|---:|
| 00–09 | Checkerboards | 0/10 | 0/10 | 0/10 |
| 10–19 | Ellipses | 9/10 | 10/10 | 10/10 |
| 20–29 | Spirals | 0/10 | 2/10 | 0/10 |
| 30–39 | Roses | 0/10 | 0/10 | 0/10 |
| 40–49 | Polygons | 1/10 | 6/10 | 3/10 |
| 50–59 | Stars | 0/10 | 0/10 | 0/10 |
| 60–69 | Lissajous curves | 0/10 | 0/10 | 0/10 |
| 70–79 | Waves | 0/10 | 1/10 | 1/10 |
| 80–89 | Gaussian rings | 2/10 | 8/10 | 5/10 |
| 90–99 | Superellipses | 9/10 | 9/10 | 9/10 |
| **Total** | **All families** | **21/100** | **36/100** | **28/100** |

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

### EXP-002 — Transformer backbone

**Question.** Does replacing the MLP with a similarly sized Transformer improve distribution fitting under the same training settings?

**Method.** Use four Transformer encoder blocks with 64-dimensional tokens, four attention heads, a 256-dimensional feed-forward network, SiLU, pre-LayerNorm, and no dropout. Each point has position, time, and class tokens; the position token predicts velocity. Parameter count: **205,954**, versus **214,850** for the MLP. This run used PyTorch attention; the optional handwritten attention implementation was added afterward.

**Configuration.** Same dataset, 15,000 updates, batch 2,048, AdamW, LR 0.001, 500-step warmup, cosine decay, seed 42, MPS device, and evaluation settings as EXP-001. Sampling remains 100 Heun steps and 2,000 points per class. Run date: 2026-09-30. Measured duration: **2,606.0 s (43.4 min)** including training progress sampling/plots, excluding final sampling/evaluation; approximately **6.1×** the baseline duration, not an isolated training-throughput measurement.

![EXP-002: Transformer particle trajectories](experiments/002-transformer/inference.gif)

**Result.** Validation MSE fell from **1.9254 to 1.1243**. Mean SW1 improved from the MLP's **0.0340 to 0.0315** (about **7.4%** lower), and passing classes increased from **21 to 36/100**. Passing counts by family: ellipses 10/10, spirals 2/10, polygons 6/10, waves 1/10, Gaussian rings 8/10, and superellipses 9/10; all other families 0/10.

**Finding.** This configuration improves the combined fit-check pass count, but 64 classes still fail and runtime is substantially higher. Its higher validation MSE does not imply worse generated distributions; validation sets differ between architectures, and flow loss is not the final fit criterion. This single-seed comparison does not establish that Transformers are generally better.

[Target vs. generated](experiments/002-transformer/comparison.png) · [Per-class results](experiments/002-transformer/quality.csv) · [Exact configuration](experiments/002-transformer/config.json) · [Metrics](experiments/002-transformer/metrics.json) · [Acceptance thresholds](experiments/002-transformer/quality_report.json)

### EXP-003 — CNN backbone

**Question.** Can a similarly sized convolutional backbone improve fitting under the same training settings?

**Method.** Project the same Fourier position/time features and class embedding into a three-position sequence. Apply four Conv1d layers with 128 channels, kernel size 3, zero padding, and SiLU, then flatten and predict 2D velocity. Convolution operates within each point's feature sequence, never across batch points or a raster image. No dropout or batch normalization. Parameter count: **207,810**.

**Configuration.** Same training and sampling settings as EXP-001/002: 100 classes, 15,000 updates, batch 2,048, AdamW, LR 0.001, 500-step warmup, cosine decay, seed 42, MPS, 100 Heun steps, and 2,000 points per class. Launch with `--model cnn --cnn-channels 128 --num-layers 4`. Measured duration: **3,979.3 s (66.3 min)** including training progress sampling/plots, excluding final sampling/evaluation; **9.4×** MLP and **1.5×** Transformer. These are observed run durations, not isolated throughput measurements.

![EXP-003: CNN particle trajectories](experiments/003-cnn/inference.gif)

**Result.** Validation MSE fell from **1.6418 to 1.1062**. Mean SW1 was **0.0322**, compared with **0.0340** for MLP and **0.0315** for Transformer. **28/100 classes passed**, versus 21 for MLP and 36 for Transformer. Passing counts: ellipses 10/10, polygons 3/10, waves 1/10, Gaussian rings 5/10, and superellipses 9/10; all other families 0/10.

**Finding.** CNN falls between MLP and Transformer on both mean SW1 and passing classes, while taking the longest in these runs. It does not outperform the Transformer on these fit metrics, and 72 classes still fail. Conclusions are limited to this configuration and single seed.

[Target vs. generated](experiments/003-cnn/comparison.png) · [Per-class results](experiments/003-cnn/quality.csv) · [Exact configuration](experiments/003-cnn/config.json) · [Metrics](experiments/003-cnn/metrics.json) · [Acceptance thresholds](experiments/003-cnn/quality_report.json)

## Evaluation protocol

Each class must pass **all four checks**: sliced Wasserstein-1 (global distribution), nearest-distance precision (proximity to target support), coverage (missing regions), and multiscale grid JS divergence (local density). Thresholds are calibrated against independent target samples: at most 1,000 points per class, five calibration draws, distance tolerance ×1.5, and probability slack 0.05. These are heuristic acceptance checks, not proof of distributional equivalence.

Keep the dataset version, evaluation settings, sampling budget, and seeds fixed when comparing experiments. Report method changes and compute budgets explicitly. These are single-seed results. Architecture-dependent initialization consumes different random draws, so training streams and validation sets are not identical across models; validation MSE is not a paired comparison. Final sampling is re-seeded independently of training.

## Reproduce

Use a Python environment with [requirements.txt](requirements.txt) installed. Run from the repository root:

```bash
python run_train.py --model mlp --device mps --num-classes 100 \
  --hidden-dim 256 --num-layers 4 --batch-size 2048 \
  --steps 15000 --warmup-steps 500 --learning-rate 0.001 --seed 42 \
  --sample-points 2000 --sample-steps 100 \
  --output-dir outputs/exp001_reproduction \
  --save-progress --save-every 1000 --log-every 500 \
  --inference-animation --animation-points 300 --inference-gif-duration-ms 50

python run_evaluate.py --output-dir outputs/exp001_reproduction
```

Use `--device cpu` or `--device cuda` on other hardware. Full checkpoints and generated files stay in the ignored `outputs/` directory. Selected results under `experiments/` are kept for GitHub. `progress.gif` shows training evolution; `inference.gif` shows particle movement through one fixed model.

## Swap models

Available backbones: `--model mlp`, `--model transformer`, and `--model cnn`.
The CNN uses four width-three, zero-padded Conv1d layers with SiLU over a
three-position sequence of position/time/class features. It never convolves
across batch points. `--cnn-channels` defaults to 128; `--num-layers` sets its
depth, and `--hidden-dim` is unused by CNN. Its default four-layer configuration
has 207,810 parameters for 100 classes. Training and sampling settings can stay
identical to the other experiments; see EXP-003 for results.

Change only `--model mlp` to `--model transformer` or `--model cnn` in the training command above, and choose a new output directory. Optimizer, data, loss, evaluation and Heun sampling are shared. The checkpoint stores the architecture, so `run_sample.py --checkpoint ...` selects the matching model automatically; old MLP checkpoints remain supported.

The Transformer uses three tokens **within each point**: Fourier position features, Fourier time features, and a class embedding. Four encoder blocks attend over these tokens; the position token predicts the 2D velocity. Points in a batch never attend to one another. Defaults: token width 64 (`--transformer-dim`), four heads (`--num-heads`), no dropout. `--hidden-dim` controls MLP hidden width or Transformer feed-forward width; `--num-layers` controls backbone depth. With the experiment settings, MLP has 214,850 parameters and Transformer has 205,954.

To add a model, implement `__init__(ModelConfig)` and `forward(points[B,2], time[B,1], labels[B]) -> velocity[B,2]`, then add its class to `MODEL_REGISTRY` in [model/registry.py](model/registry.py). No changes to the trainer or sampler are needed.

## Add an experiment

Append one table row and one short entry: **question → method/change → configuration → results → inference GIF → finding**. Save selected artifacts in `experiments/004-<name>/` with the exact config and evaluation report. Retain failures and keep acceptance thresholds unchanged across comparisons.
