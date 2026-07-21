# Flow Matching Standardization

## Old Custom Behavior

The trainer previously sampled a continuous timestep uniformly between
`train.min_timestep` and `train.max_timestep`, converted it to
`sigma = timestep / 1000`, then constructed:

```python
noisy = (1 - sigma) * target + sigma * noise
training_target = noise - target
```

That formula was mathematically self-consistent. The mismatch with the active
DiffSynth Wan SFT path was primarily the sigma sampling distribution and the
absence of scheduler timestep weighting.

The previous behavior is preserved as an explicit ablation mode:

```json
"flow_matching": {
  "mode": "uniform_linear",
  "min_timestep": 0.0,
  "max_timestep": 1000.0,
  "use_training_weight": false
}
```

## Standard DiffSynth Wan Behavior

The recommended mode now follows the active repository-native DiffSynth Wan
training scheduler:

```json
"flow_matching": {
  "mode": "diffsynth_wan",
  "num_train_timesteps": 1000,
  "sigma_shift": 5.0,
  "min_timestep_boundary": 0.0,
  "max_timestep_boundary": 1.0,
  "use_training_weight": true
}
```

At startup the trainer calls:

```python
pipe.scheduler.set_timesteps(num_train_timesteps, training=True, shift=sigma_shift)
```

The training and validation paths then use the scheduler's own
`add_noise`, `training_target`, and `training_weight` methods. The trainer does
not manually reimplement the Wan shift equation.

## What Changed

- Added explicit `train.flow_matching` configuration with `diffsynth_wan` and
  `uniform_linear` modes.
- Initialized the Wan training scheduler once before the training loop.
- Added scheduler invariant checks for timestep count, sigma count, finite
  values, training weights, and `timestep ~= sigma * 1000`.
- Replaced separate training and validation noise code with one shared
  `build_flow_matching_batch()` helper.
- Made validation deterministic under scheduler-index sampling by deriving
  noise and timestep index from `validation.seed + sample_index`.
- Added weighted and unweighted loss accounting.
- Saved the effective flow-matching config in copied config files, DeepSpeed
  client state, and lightweight checkpoint metadata.
- Added full-resume compatibility checks for mode, timestep count, and sigma
  shift.

## What Remains Custom

These parts remain project-specific and were not rewritten:

- local-only and dual-context conditioning;
- per-keyframe local prompt routing;
- custom keyframe cross-attention;
- 16-slot latent representation;
- cached VAE, text and CLIP feature loading;
- first-frame conditioning through latent slot replacement;
- LoRA/full parameter grouping and optimizer policy.

## Velocity Target

The target remains `noise - target`. DiffSynth Wan's
`FlowMatchSFTLoss` uses `scheduler.training_target()`, and the active scheduler
returns this same velocity target. The change is therefore not a target
definition change.

## Why Sampling And Weighting Changed

The standard mode changes how sigmas are sampled and optionally applies the
scheduler's empirical training weights. This aligns SFT with the repository's
active Wan implementation. It does not prove better visual quality by itself;
controlled comparison against `uniform_linear` is still required.

## Exp4 Migration Example

For the current Exp4-style config, keep the dataset/cache/optimizer settings
unchanged and add:

```json
"flow_matching": {
  "mode": "diffsynth_wan",
  "num_train_timesteps": 1000,
  "sigma_shift": 5.0,
  "min_timestep_boundary": 0.0,
  "max_timestep_boundary": 1.0,
  "use_training_weight": true
}
```

The legacy ablation equivalent is:

```json
"flow_matching": {
  "mode": "uniform_linear",
  "min_timestep": 0.0,
  "max_timestep": 1000.0,
  "use_training_weight": false
}
```

## TensorBoard Metrics

- `train/loss`: actual optimized loss.
- `train/loss_unweighted`: masked MSE before timestep weighting.
- `train/loss_weighted`: masked MSE multiplied by scheduler weight.
- `train/timestep`: sampled model timestep.
- `train/timestep_index`: scheduler index in `diffsynth_wan`, `-1` in
  `uniform_linear`.
- `train/sigma`: sigma corresponding to the sampled timestep.
- `train/timestep_weight`: scalar scheduler weight, or `1` when disabled.
- `val/loss_unweighted`: aggregate validation masked MSE.
- `val/loss_weighted`: aggregate validation weighted masked MSE.
- `val/loss`: alias for weighted validation loss.
- `val/timestep_weight_mean`: mean validation timestep weight.
- `val/loss_by_slot/slot_*`: unweighted per-slot MSE.
- `val/loss_by_sigma_bin/bin_*`: unweighted MSE binned by actual sigma.
- `val/loss_by_task/*`: unweighted MSE grouped by task prefix.

## Known Limitations

- DiffSynth's `FlowMatchSFTLoss` samples one timestep per micro-batch. This
  trainer currently enforces micro-batch size 1, so the behavior matches that
  assumption.
- Validation loss is deterministic and useful for regression, but it is not a
  substitute for fixed-seed video generation checks.
- We do not checkpoint scheduler tensors because they are deterministic tables
  reconstructed from the effective flow-matching config.
- Weights-only resume may intentionally start a new flow-matching ablation and
  prints a warning instead of refusing to load.
