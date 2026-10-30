# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Deterministic world model for a PyTorch reproduction of SimPLe on Pong, for a reinforcement learning class final project. Legibility and informative comments matter as much as function, because the code is graded and read by teammates.

`SPEC.md` is authoritative and read-only. It changes as the project progresses, so re-read it at the start of each task rather than relying on anything here or remembered from earlier. Build only what it currently covers.

## Where things live

- `SPEC.md`: what to build.
- `teammates/`: teammates' env and data code, for reference. `RL_Final_Project.ipynb` is a local copy of that work and `rl_final_project.py` is its Colab `.py` export, with the same code. The `.py` is easier to read and grep.
- `docs/paper.md`: the paper. Appendix C is the architecture source. `docs/arXiv-1903.00374v5/` is the LaTeX source.
- `reference/tensor2tensor/`: the authors' TF1 code, for reference only. Start at `tensor2tensor/models/video/basic_deterministic.py`. The closest hyperparameters are `next_frame_basic_stochastic_discrete` in `basic_stochastic.py`, which match Appendix C (`hidden_size=96`, `filter_double_steps=3`, `dropout=0.15`). It is a bare gitlink with no `.gitmodules`, so a fresh clone will not contain it.
- `old/`: the superseded spec, plan, and rules from before the scope was pared back. Commands and files named there (`train_wm.py`, `configs/`, `tests/`, `DEVIATIONS.md`, phases P0-P4) do not exist. It is still useful for its paper section citations.

## Delivering code

The final destination is a shared Colab notebook that this repo cannot reach. New work goes in a Python file or notebook of its own outside `teammates/`, and the user pastes it into Colab by hand. So:

- Keep the code self-contained and easy to paste, with the model class and its unit tests separable into different cells, as the spec requires.
- Write object-oriented code, per the spec.
- Assume teammates' names (`EnvConfig`, `make_env`, `ReplayBuffer`, `to_torch`) are already defined in the Colab, and don't copy them into new files.

## Commands

There is no build, lint, or test tooling, and no requirements file. In Colab, the notebook's first code cell installs `gymnasium[atari]` and `opencv-python-headless`, and PyTorch comes from the runtime. The local `python3` has none of these packages, so running anything locally needs an environment with `torch` set up first.

## Testing

Every piece of model code gets basic, relevant unit tests, and they should be run before the work is called done. Say so plainly if they could not be run. At a minimum:

- **Smoke tests:** a forward and backward pass run, and output shapes are correct for the frame and reward heads.
- **Overfitting:** the model overfits a small synthetic batch, with loss falling close to zero, to show that it can learn.

Use synthetic tensors shaped like the buffer's output, so that the tests need only `torch` and not the Atari env or a collected dataset.

## Architecture

Teammates' code is a pipeline that feeds the world model:

`EnvConfig` -> `make_env` (`SimPLeEnv`, `FrameStack`) -> `collect(env, policy, n, buffer)` -> `ReplayBuffer` -> world model

Shapes flow from the env rather than from constants: `ReplayBuffer.from_env` reads them from `make_env(ENV_CFG, stack=True)`. Do the same in the model where practical, and don't change `EnvConfig` values, because saved buffers are rejected on a config mismatch.

### Buffer-to-model interface

`buffer.sample_batch(B, rng)` returns NumPy `(stack, action, next_frame, reward)`:

| Item | Shape | Note |
| --- | --- | --- |
| `stack` | `(B, 4, 3, 105, 80)` uint8 | Oldest frame first. Flatten to 12 channels for the model. |
| `action` | `(B,)` int64 | `Discrete(6)` |
| `next_frame` | `(B, 3, 105, 80)` uint8 | |
| `reward` | `(B,)` float32 | Values in {-1, 0, 1}. Add 1 to get a class index. |

`to_torch` scales `next_frame` to float [0, 1], but the 256-way per-pixel output needs integer targets in 0-255, so take frame targets before that scaling. Never train on `split="holdout"`.

### Model wiring

Appendix C gives layer sizes but not the wiring. T2T's `NextFrameBasicDeterministic.next_frame` shows how it fits together:

- **Spatial sizes:** the encoder goes 105x80 -> 53x40 -> 27x20 -> 14x10 -> 7x5 -> 4x3 -> 2x2. The odd sizes mean decoder outputs overshoot, so T2T crops each one to its skip's shape before adding.
- **Residuals:** additive, followed by layer norm. Upscale `i` adds the output of downscale `6 - i`, where downscale 0 is the input dense layer. This is the pairing that makes the widths in the spec's table match (768, 768, 768, 384, 192, 96).
- **Output:** "Output frame dense 768" is 3 channels x 256 values per pixel. T2T emits these as logits.

Points where the sources disagree or go beyond the paper. Check `SPEC.md` first, and if it doesn't settle one, ask before picking:

- **Action injection:** Fig 2 says the decoder is multiplied by the embedded action. T2T defaults to `multi_additive` (sigmoid multiply plus add) and also injects once at the bottleneck.
- **Reward head:** T2T concatenates the spatial means of the bottleneck (before the middle convs) and the final decoder features, then applies dense 128.
- **Input and extras:** T2T standardizes each image rather than scaling to [0, 1], and adds timing signals that the paper doesn't mention.
- **Middle convs:** T2T uses `residual_dropout` (0.5) here rather than 0.15, with a residual around the second conv.

## Rules

- Treat `SPEC.md`, `reference/`, `teammates/`, and `docs/` as read-only. Raise problems in teammates' code with the owning teammate instead of patching it.
- Don't depart from a value in `SPEC.md` without asking.
- If the spec is ambiguous, stop and ask rather than guess.
- Cite the paper section in a comment for every non-obvious design choice.
- Never edit tests to make them pass.
