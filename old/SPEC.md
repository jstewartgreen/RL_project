# SPEC: SimPLe world model (Pong)

**Source tags:** [§N] paper section · [App X] appendix · [Fig N] figure/caption · [T2T] reference code
(tensor2tensor/rl) · [OURS] group decision · [TBD] unresolved: stop and ask before implementing

**Phase tags:** [P0]–[P4], see PLAN.md. Only implement sections tagged with the current phase or earlier.
Later-phase sections are here so earlier code leaves room for them, not so they get built early.

---

## 0. Scope
- This lane builds env' (the world model), its training, and the batched simulated-env wrapper.
- PPO, policy evaluation, and the outer SimPLe loop belong to teammates; §7 is the contract.
- The env wrapper and replay buffer already exist as teammates' code in `teammates/` (read-only).
  §1–2 describe what that code must provide; verify it, don't rebuild it. Mismatches go to the
  owning teammate.
- Target game: Pong only. Runtime: shared Colab (see CLAUDE.md).
- "Deterministic model" = the App C frame predictor with no latent (inference net, bits, and bit
  predictor all absent) [Fig 2 caption]. NOT the 64-filter model described in §4 ¶2. [OURS]

## 1. Environment & preprocessing [P0, teammates' code: verify only]
- Frameskip 4 (each action repeated 4 times) [§6]
- Frames downscaled ×2: 210×160×3 → 105×80×3 [§6, App C]
- Observation = 4 consecutive frames stacked → 105×80×12 [§3, Fig 2]
- Deterministic ALE: no sticky actions (`repeat_action_probability=0`). Sticky actions appear only
  in the §6.3 experiment. [§6, §6.3]
- Action set: [TBD: minimal (6) vs full (18); check T2T]
- Downscale method (area/bilinear/nearest): [TBD, check T2T]
- Rewards are 3 classes {-1, 0, +1}, encoded as class index r+1 [App C reward head; encoding OURS]

## 2. Data [P0, teammates' buffer: verify; adapt only what's missing]
- Buffer D holds real-env trajectories: frames (uint8), actions, rewards, episode boundaries [OURS]
- Anything the world model needs that the buffer lacks (sequence sampling, start-state sampling,
  save/load) goes in our own adapter module, not in `teammates/` [OURS]
- Paper budget: initial random rollouts, then 6,400 real interactions per loop iteration;
  6,400 × 16 = 102,400 total [§6]
- Training examples: (4 input frames, action) → (next frame, reward) [§4]
- For scheduled sampling [P2], training samples short consecutive sequences, not single transitions
  [OURS, sequence length TBD from T2T]
- env' reset states are sampled uniformly from D [§5]
- D must be saveable/loadable from a configurable path [OURS]

## 3. Architecture (authoritative: App C)

### 3.1 Frame predictor [P1]
| Layer | Outputs | Details |
|---|---|---|
| Input frame dense | 96 | per-pixel dense over the 12 stacked channels |
| Downscale conv 1–6 | 192, 384, 768, 768, 768, 768 | 4×4, stride 2 |
| Middle conv 1–2 | 768, 768 | 3×3, stride 1 |
| Upscale deconv 1–6 | 768, 768, 768, 384, 192, 96 | 4×4, stride 2 |
| Output frame dense | 768 | = 3 channels × 256 logits per pixel |

- Residual/skip connections from each downscale layer to the matching upscale layer [App C]
- Dropout 0.15 before every conv/deconv; layer norm after; ReLU except output layers [App C]
- Spatial sizes go 105×80 → 53×40 → 27×20 → 14×10 → 7×5 → 4×3 → 2×2 [Fig 2]; decoder outputs must
  be cropped or padded to match the skip shapes on the way up [OURS]
- Input pixel scaling/encoding before the dense layer: [TBD, check T2T]
- Expected size of the full model (with latent parts): ~74M params [App C]

### 3.2 Action conditioning [P1]
- Action one-hot → learned embedding (768) [§4, App C]
- Embedded action multiplied channel-wise into the output of each decoder layer [Fig 2 caption]
- How the 768-d embedding meets decoder layers of width 384/192/96 (per-layer projection?): [TBD, T2T]

### 3.3 Reward head [P1]
- Hidden dense 128 → output dense 3 → softmax [App C]
- Attachment point (bottleneck vs. elsewhere): [TBD, check T2T]

### 3.4 Stochastic discrete latent [P3]: do not implement before P3
- Inference net, training only: conv 128 (8×8, stride 4) → conv 512 (8×8, stride 4) [App C];
  input = the 4 input frames + the target next frame [§4]
- 128-d latent; uniform noise added, then discretized to bits; straight-through gradient
  (backprop bypasses the discretization) [§4, Fig 2, Kaiser & Bengio 2018]
- Dropout on bits after discretization [§4]; rate: [TBD]
- Bit predictor: embedding 128 → LSTM 128 → dense 256; predicts 128 bits autoregressively,
  8 bits at a time (16 steps; 256 = 2^8 classes per step) [App C]
- At inference, bits come from the bit predictor, not from a prior; no KL term [§4]
- Where the bits enter the frame predictor: [TBD; Fig 2 suggests the bottleneck; check T2T]
- P1 requirement: the decoder must have an obvious place to accept this input later

## 4. Losses
- Frame [P1]: per-pixel 256-way softmax cross-entropy, clipped per pixel as max(loss, 0.03),
  then averaged. A pixel predicted with >97% confidence contributes no gradient. [§4]
- Reward [P1]: cross-entropy over 3 classes
- Bit predictor [P3]: cross-entropy against the inference net's bits
- Relative loss weights: [TBD, check T2T]

## 5. Training
- Scheduled sampling [P2]: during training, randomly replace input frames with the model's own
  previous predictions; mixing probability rises linearly to 100% around the middle of loop
  iteration 1 [§4]
- World model steps [P4]: 45K in iteration 1, 15K in each later iteration [§5]; the best results
  used ×5 longer training [App A]
- Optimizer, LR, schedule, batch size: [TBD, not in paper; check T2T]
- Checkpoint every loop iteration to a configurable path; training must resume from the last
  checkpoint [OURS]
- Log scalars (losses, accuracies, eval metrics) to a file (e.g. metrics.jsonl) [OURS]

## 6. Scaled-down config [P4] [OURS]
- Widths, steps, and batch size chosen to fit the Colab GPU; rationale recorded in DEVIATIONS.md
- Smoke config: tiny widths, runs on CPU in under 2 minutes; used for agent self-checks [OURS]

## 7. Interface contract with policy code [P2]
- Batched env' with 16 parallel envs [§5]
- `reset(start_states)` takes 4-frame stacks sampled from D → obs
- `step(actions)` → (obs, reward, done)
- Rollouts are short, N=50 steps, then reset from D [§5]. env' does not predict episode end;
  whether env' or the PPO side enforces N: [OURS, agree with team]
- Value-function bootstrap at the rollout's last step happens on the PPO side [§5]
- Shapes, dtypes, device, and whether obs are uint8 frames or model logits: [OURS, agree with team;
  record in interface.py]

## 8. Evaluation & success criteria
- [P0] Shape/dtype tests for preprocessing and buffer
- [P1] Overfits one batch to near-zero loss; reward head reaches 100% accuracy on that batch
- [P2] k-step autoregressive prediction error vs. real frames, k up to 50; Pong should be
  pixel-perfect over short horizons [App B]; correct reward predictions per sequence [§4]
- [P3] Gradients flow through the straight-through bits; sampled rollouts differ across seeds
- End-to-end reference scores, Pong at 100K interactions, mean (std) over 5 runs [App D]:
  - Ours, deterministic: -17.4 (5.2)
  - Ours, SD: 5.2 (9.7)
  - Ours, SD long (headline SimPLe): 12.8 (17.2)
  - Random -20.4 · Human 15.0

## 9. Decisions, ambiguities & conflicts
- **Architecture widths:** §4 ¶2 describes a 64-filter conv + dense model; App C lists the 96→768
  model. Using App C. [OURS]
- **Dropout:** 0.2 [§4, the 64-filter model] vs 0.15 [App C]. Using 0.15. [OURS]
- **PPO γ:** 0.95 [§5] vs 0.99 "unless specified" [§6.4, App A]. Policy-side; teammates decide.
- **Unspecified in the paper, resolve from T2T, then move into the body above:** action set,
  downscale method, input pixel encoding, action-embedding projection, reward-head placement,
  latent injection point, bit dropout rate, loss weights, optimizer/LR/batch, scheduled-sampling
  sequence length.
