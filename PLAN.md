# PLAN: world model lane

**Current phase: P0**  (keep in sync with CLAUDE.md)

Rules: work only on the current phase. Each phase is done when every exit criterion is checked.
Leave the extension points listed; don't build what they're for.

---

## P0: Integrate & verify teammates' env and buffer ⬜
The env wrapper and replay buffer are teammates' code in `teammates/` (read-only). Do not rebuild them.

**Deliverables**
- [ ] Audit report (docs/AUDIT.md): teammates' env and buffer checked against SPEC §1–2, each item
      marked matches / differs / not covered, with file and line
- [ ] Any mismatch raised with the owning teammate, not patched in `teammates/`
- [ ] Thin adapter in our own code (e.g. `wm_data.py`) only for what the buffer lacks: sequence
      sampling for P2, start-state sampling for env' resets, save/load
- [ ] `interface.py` stub for env' (SPEC §7), agreed with teammates

**Exit criteria**
- [ ] Tests (ours, in `tests/`) confirm: obs shape 105×80×12 uint8, frame order in the stack,
      frameskip 4, no sticky actions, rewards in {-1, 0, +1}, no frame stacks spanning an episode reset
- [ ] Buffer data round-trips through save/load unchanged
- [ ] SPEC §9 TBDs needed for P1 resolved from T2T (pixel encoding, action projection,
      reward-head placement)

**Out of scope:** any model code; any edits inside `teammates/`

---

## P1: Deterministic frame predictor ⬜
**Deliverables**
- [ ] Encoder / middle / decoder with skips (SPEC §3.1)
- [ ] Action conditioning (SPEC §3.2)
- [ ] Reward head (SPEC §3.3)
- [ ] Clipped per-pixel loss + reward loss (SPEC §4)
- [ ] Single-step training loop, checkpointing, metrics.jsonl logging
- [ ] Smoke config (CPU, under 2 min) and full-width config

**Exit criteria**
- [ ] Overfits one batch to near-zero loss; reward accuracy 100% on that batch
- [ ] Param count printed for each config
- [ ] Smoke config runs end to end on CPU
- [ ] Imports and runs in Colab via the `%%writefile` cell

**Must leave:** a clear place in the decoder to accept latent bits (P3)
**Out of scope:** latent, inference net, bit predictor, scheduled sampling, multi-step rollout

---

## P2: Multi-step & env' interface ⬜ ← teammate integration milestone
**Deliverables**
- [ ] Scheduled sampling over short training sequences (SPEC §5)
- [ ] Autoregressive k-step rollout
- [ ] Batched `SimulatedEnv` implementing `interface.py`; reset from D (SPEC §7)

**Exit criteria**
- [ ] k-step prediction error (k up to 50) logged against real frames
- [ ] Teammates' PPO runs against env' with no changes on their side
- [ ] All P1 tests still pass

---

## P3: Stochastic discrete latent ⬜
**Deliverables**
- [ ] Inference net (SPEC §3.4)
- [ ] Noise + straight-through discretization + bit dropout
- [ ] LSTM bit predictor, autoregressive sampling at inference
- [ ] Bit-predictor loss

**Exit criteria**
- [ ] Gradient-flow test through the bits passes
- [ ] Sampled rollouts differ across seeds from the same start state
- [ ] P1/P2 tests still pass with the latent disabled

---

## P4: Scaled training on Colab ⬜
**Deliverables**
- [ ] Scaled-down config fitting the Colab GPU; rationale in DEVIATIONS.md (SPEC §6)
- [ ] Training resumable across loop iterations and Colab disconnects

**Exit criteria**
- [ ] [OURS: team defines the target, e.g. Pong score vs. SPEC §8 reference numbers]
