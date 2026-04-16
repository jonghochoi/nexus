# 📋 Confluence & MLflow: Role Separation Guide

---

## 🧭 The Core Principle

> These two tools answer fundamentally different questions.
> **Never mix them.** The moment you paste raw numbers into Confluence or write
> interpretations into MLflow, both tools become harder to use.

| | 📊 MLflow | 📝 Confluence |
|---|---|---|
| **Answers** | *"What happened?"* | *"Why did it happen, and what do we do?"* |
| **Contains** | Numbers, curves, checkpoints | Reasoning, decisions, hypotheses |
| **Written by** | Training code (automatic) | Humans (intentional) |
| **Updated** | Every epoch | Every experiment group |

---

## 📊 What Goes in MLflow

> MLflow is the **single source of truth for all quantitative data**.
> Nothing gets re-typed into Confluence — Confluence only links to MLflow.

| Category | Examples |
|---|---|
| 📈 **Metrics (time-series)** | `losses/actor_loss`, `performance/RLTrainFPS`, `info/kl` |
| ⚙️ **Hyperparameters** | `lr`, `gamma`, `e_clip`, reward weights |
| 🏷️ **Reproducibility tags** | `seed`, `isaac_lab_version`, `physx_solver`, `researcher` |
| 💾 **Artifacts** | `checkpoint.pt`, `reward_fn.py`, `env_cfg.yaml` |
| ℹ️ **Run metadata** | start time, duration, GPU used, status |
| 🔗 **Sim-to-Real linkage** | `sim_run_id` tag on real-robot eval runs |

> ⚠️ **Hard rule:** If it's a number produced during training, it lives in MLflow **only**.

---

## 📝 What Goes in Confluence

> Confluence captures **human judgment** — the context, interpretation, and decisions that MLflow cannot store.

| Category | Examples |
|---|---|
| 💡 **Hypothesis** | *"We believe fingertip tactile reward will reduce slip events by >20%"* |
| 🎯 **Experiment intent** | Why this ablation was designed, what question it answers |
| 🔍 **Interpretation** | *"v2 improved stability but reward collapsed at high rotation angles"* |
| 💥 **Failure analysis** | Edge cases observed, suspected root causes |
| ✅ **Decisions** | What the team agreed to do next, and why |
| 🔗 **MLflow links** | Direct URL to the run(s) being discussed — no numbers copied |

> ⚠️ **Hard rule:** If it's a sentence that requires a human to write, it lives in Confluence **only**.

---

## 🗂️ Confluence Page Structure

```
[Space] Robot Hand — Dexterous Manipulation Research
│
├── 📋 Project Overview
│   ├── Research goals and milestones
│   └── Infrastructure guide (MLflow access, naming conventions)
│
├── 🧪 Experiment Log                    ← one page per experiment group
│   ├── [2025-Q2] Reward Shaping Search
│   ├── [2025-Q2] Tactile Feedback Integration
│   └── [2025-Q3] Sim-to-Real Transfer
│
├── 🔍 Ablation Studies
│   ├── Contact Force Threshold Sweep
│   └── DOF Masking Strategy Comparison
│
├── 📌 Decision Log                      ← team decisions only, no raw data
│
└── 📚 Reference
    ├── Robot Hand kinematic constraints
    └── PhysX contact model behavior notes
```

---

## 📝 Experiment Page Template

> Use this template for every experiment group.
> **Write the hypothesis before running the experiment.**

```markdown
# [Experiment Name]

## 1. Objective & Hypothesis
- Problem: [failure mode or limitation being addressed]
- Hypothesis: [specific, falsifiable prediction with a threshold]

## 2. MLflow Runs
| Run Name | MLflow Link | Owner | Status |
|---|---|---|---|
| ppo_tactile_v1 | [link](http://mlflow-server:5000/...) | @name | Done |

→ All metrics and curves: open the MLflow link above.
   Do NOT copy numbers into this page.

## 3. Key Findings (interpretation only — no raw numbers)
## 4. Failure Analysis / Edge Cases
## 5. Next Steps
## 6. Decision
```

---

## 📌 Decision Log Template

> A separate page listing only final team decisions — no experiment details.
> Allows anyone to reconstruct the project direction without reading every page.

| Date | Decision | Rationale | Source | Owner |
|---|---|---|---|---|
| 2025-04-16 | Adopt tactile reward (v2, reweight) | Slip reduction confirmed despite edge case | [link] | Team |
| 2025-04-10 | Defer PPO → SAC migration | SAC sample cost too high at current DOF | [link] | @name |

---

## 🚦 Operational Rules

### ✅ Do

| Rule | Why |
|---|---|
| Write the Confluence page **before** the experiment runs | Prevents post-hoc rationalization |
| Link to MLflow runs, never copy numbers | Numbers in Confluence go stale |
| Close **every** experiment page — including failures | Prevents repeating the same mistake |
| Tag every MLflow run before training starts | Enables reproducibility and fair comparison |
| Add `sim_run_id` on all real-robot eval runs | Required for Sim-to-Real failure tracing |

### ❌ Don't

| Anti-pattern | Why it fails |
|---|---|
| Pasting metric tables into Confluence | Creates two sources of truth that diverge |
| Writing interpretation as MLflow run description | Gets buried; not visible to the team |
| Skipping Confluence page for "quick" experiments | Quick experiments produce uninterpretable results later |
| Starting training before writing the hypothesis | Makes findings indistinguishable from noise |
| Leaving failed experiment pages empty | Team will repeat the same mistake |

---

## 🏷️ Required MLflow Tags

> ⚠️ Isaac Lab / PhysX results are non-deterministic without these.
> Set them before every run — no exceptions.

```bash
--tags \
  researcher=<name> \
  seed=<int> \
  isaac_lab_version=<x.x.x> \
  physx_solver=<TGS|PGS> \
  task=<task_name> \
  hardware=robot_22dof
```

For real-robot evaluation runs, **additionally** add:

```bash
--tags sim_run_id=<upstream_training_run_id>
```
