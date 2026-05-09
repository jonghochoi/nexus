01_EXPERIMENT_STANDARD.md
=========================
Team experiment management standard for NEXUS. Every team member must follow
these rules to keep the shared MLflow server useful as a collective knowledge
base.

> **One-liner:** MLflow = numbers. Confluence = judgment. Names = rules. Hypothesis = before you run.

---

## Table of Contents

- [Tool role separation](#tool-role-separation)
- [Experiment structure](#experiment-structure)
- [Run naming convention](#run-naming-convention)
- [Tags reference](#tags-reference)
- [Run structure — standalone vs parent/child](#run-structure--standalone-vs-parentchild)
- [Experiment lifecycle](#experiment-lifecycle)
- [Failed run policy](#failed-run-policy)
- [Sim-to-Real connection rule](#sim-to-real-connection-rule)
- [What NOT to do](#what-not-to-do)
- [Pre / post checklist](#pre--post-checklist)
- [Confluence page template](#confluence-page-template)

---

## Tool role separation

> **Never mix the two tools' roles.**

| | 📊 MLflow | 📝 Confluence |
|---|---|---|
| **Question** | *"What numbers came out?"* | *"Why did that happen, and what next?"* |
| **Stores** | Numbers, curves, checkpoints | Hypotheses, decisions, failure analyses |
| **Written by** | Training code (automated) | People (intentionally) |
| **Updated** | Every epoch | Per experiment group |

**MLflow records:** metrics (time-series), hyperparameters, run tags, artifacts (`best.pth`, `last.pth`), run metadata, `sim_run_id` links.

**Confluence records:** hypotheses (written *before* the run), result interpretations, failure analyses, team decisions, MLflow links (never paste raw numbers).

---

## Experiment structure

An experiment is **"a group of runs you want to compare together."** Runs in different experiments cannot be compared side-by-side in the MLflow UI.

| Experiment name | Purpose |
|---|---|
| `baseline_ppo` | PPO baseline runs, seed diversity |
| `reward_shaping` | Reward function structure / weight search |
| `ablation_contact` | Contact-force reward ablations |
| `ablation_tactile` | Tactile / fingertip obs ablations |
| `sim2real_transfer` | Sim policy → real robot transfer |
| `real_robot_eval` | Physical robot evaluation (requires `sim_run_id`) |

**Creating a new experiment** — only when a completely new research direction does not fit any existing experiment. Announce in the team channel, get consensus, then create. Name format: `<purpose>` or `<purpose>_<scope>` (lowercase, underscores).

---

## Run naming convention

### ── Format

```
<researcher>_<method>_<key_variable>_<version>
```

| Part | Description | Example |
|---|---|---|
| `<researcher>` | Short name, unique within the team | `kim`, `lee`, `park` |
| `<method>` | Algorithm or approach | `ppo`, `tactile`, `contact` |
| `<key_variable>` | What changed in this run | `contact0.3`, `lr1e4`, `seed42` |
| `<version>` | Disambiguates reruns with the same config | `v1`, `v2`, `v3` |

### ── Examples

```
✅  kim_ppo_seed42_v1
    lee_tactile_fingertip_v2
    park_contact_weight0.5_v1

❌  test_run              ← no indication of what was tested
    ppo_1234              ← no indication of who ran it
    final_real            ← no version suffix
```

**Versioning rules:** always start at `v1`; increment for reruns with the same goal (never delete the old one); use a new method segment when the approach changes significantly.

---

## Tags reference

### ── Required

| Tag | Description | Example |
|---|---|---|
| `experiment` | Experiment group name (auto-injected via `--experiment`) | `robot_hand_rl` |

### ── Strongly recommended

| Tag | Description | Example |
|---|---|---|
| `researcher` | Who ran the experiment | `kim` |
| `task` | Task name | `in_hand_reorientation` |
| `hand` | Hardware identifier | `robot_22dof` |
| `method` | Core methodology | `ppo`, `tactile` |

### ── Optional

| Tag | When to use | Example value |
|---|---|---|
| `component` | Explicitly marks which module changed | `reward`, `obs`, `network` |
| `ablation_target` | Marks this as an ablation study | `contact_weight` |
| `baseline_run_id` | Points to the reference run | `abc123def456` |
| `sim_run_id` | Links a real-robot run to its upstream sim run | `xyz789abc` |
| `fail_reason` | On failure — short summary of cause | `OOM: batch_size too large` |

### ── Auto-set tags (Pipeline A only)

`git_commit`, `git_dirty` — recorded automatically at run start by `MLflowLogger`.
When `git_dirty=true` the full diff is saved as `artifacts/git/git_patch.html`.

---

## Run structure — standalone vs parent/child

**Standalone run** — the default for almost all experiments:

```python
self.writer = make_logger(
    mode="dual",
    run_name="kim_ppo_contact0.3_v1",
    experiment_name="reward_shaping",
    ...
)
```

**Parent / child runs** — use only when sweeping a single variable:

```
✅ Use parent/child
   contact_weight sweep: [0.1, 0.3, 0.5, 0.7]
   seed replication: [42, 123, 456]
   lr sweep: [1e-3, 3e-4, 1e-4]

❌ Do NOT use parent/child
   Independent experiments by different researchers
   Experiments with fundamentally different approaches
```

With `SweepLogger` from `nexus.logger.sweep_logger`, child runs are nested under
the parent in the MLflow UI. See [`30_ADVANCED_FEATURES.md`](30_ADVANCED_FEATURES.md#1-sweeplogger--hyperparameter-sweep-management).

---

## Experiment lifecycle

```
1️⃣  Before the run
    └─ Write Confluence page (hypothesis + purpose first)
    └─ Decide experiment and run names
    └─ Verify required tags are set in agent_cfg

2️⃣  During training (Pipeline A)
    └─ make_logger() auto-logs params + tags at start
    └─ Metrics logged every step; auto-synced to central every 5 min

3️⃣  Training ends
    └─ status=FINISHED (or FAILED) set automatically
    └─ best.pth + last.pth uploaded to MLflow

4️⃣  After the run
    └─ Update Confluence page (interpretation, failure cause, next direction)
    └─ Share result in team channel (include MLflow link)
    └─ Update Decision Log if the team made a decision
```

---

## Failed run policy

> **Never delete a failed run.**

Failed experiments carry as much information as successful ones — they prevent the team from repeating the same mistakes.

| Do | Don't |
|---|---|
| Set `fail_reason` tag with a short cause summary | Delete the run from MLflow |
| Write a "Failure analysis" section in Confluence | Re-run without recording why it failed |
| Leave the run in MLflow permanently | Leave the Confluence page blank |

```
fail_reason examples:
  "OOM: num_envs=8192 exceeded GPU memory"
  "KL explosion: lr=1e-3 too high, diverged after step 0.01"
  "reward collapse: contact_weight=1.0 dominated all other terms"
```

---

## Sim-to-Real connection rule

Every real-robot eval run **must** carry a `sim_run_id` tag linking it to the upstream simulation training run. Without it, tracing a Sim-to-Real failure back to its origin is impossible.

```python
# In real_robot_eval experiment
self.writer = make_logger(
    experiment_name="real_robot_eval",
    run_name="kim_real_20250418",
    tags={
        "sim_run_id":      "abc123def456",   # ← required
        "sim_experiment":  "baseline_ppo",
        "sim_run_name":    "kim_ppo_seed42_v1",
    },
    ...
)
```

**Pipeline B:** place a `run_meta.json` file (`{"sim_run_id": "..."}`) next to the tfevents directory. Uploads to `--experiment real_robot_eval` are **blocked** if `sim_run_id` is missing. Detail: [`13_POST_UPLOAD.md`](13_POST_UPLOAD.md) §5.

---

## What NOT to do

| ❌ Prohibited action | Why |
|---|---|
| Delete a run from MLflow | Permanent loss of team experiment history |
| Paste metric values into Confluence | Numbers go stale; Confluence becomes misleading |
| Write interpretation in MLflow run description | Not visible to the team — gets buried |
| Ignore the run naming convention | Nobody can tell who ran it or what changed |
| Skip the Confluence page for "quick" tests | Results become uninterpretable later |
| Start a run without a hypothesis | You can't interpret the results objectively |
| Leave a failed run's Confluence page blank | Team will repeat the same mistake |
| Create a real-robot eval run without `sim_run_id` | Sim-to-Real failure tracing becomes impossible |
| Register a second cron on a shared GPU server | Causes duplicate metric points on central MLflow |

---

## Pre / post checklist

### ── Before starting a run

- [ ] Confluence experiment page written (hypothesis included)
- [ ] Experiment name confirmed (from existing list, or team-approved new one)
- [ ] Run name follows `<researcher>_<method>_<key_variable>_<version>` format
- [ ] `experiment` tag confirmed (auto-injected via `--experiment` arg)
- [ ] Recommended tags set (`researcher`, `task`, `hand`, `method`)
- [ ] Local MLflow server running (`bash scheduled_sync/start_local_mlflow.sh`)

### ── After training completes

- [ ] Run status shows `FINISHED` (or `FAILED`) in MLflow
- [ ] `best.pth` artifact uploaded and visible
- [ ] Confluence page updated (result interpretation, next direction)
- [ ] Result shared in team channel (with MLflow link)
- [ ] Decision Log updated if the team made a decision

### ── On failure

- [ ] `fail_reason` tag recorded
- [ ] "Failure analysis" section written in Confluence
- [ ] Run **not** deleted (**never delete**)

---

## Confluence page template

Copy this template when creating a new Confluence page for an experiment group.

```
── [Experiment name] — [one-line description] ──────────────────────

| Experiment       | Run name pattern                     | Owner | Start date | Status |
|------------------|--------------------------------------|-------|------------|--------|
| experiment_name  | researcher_method_variable_v1        | @name | YYYY-MM-DD | 🟡 Planned / 🔄 Running / ✅ Done / ❌ Failed |

── [Before the run] Hypothesis & design ────────────────────────────

> Write before training starts.

| Problem to solve | Hypothesis (measurable prediction) | Success criterion |
|------------------|------------------------------------|-------------------|
|                  |                                    | metric > target @ N steps |

Changes — one variable at a time:
| Item | Baseline | This run | Reason for change |
|------|----------|----------|--------------------|

── [After the run] Results ─────────────────────────────────────────

| Item              | Content |
|-------------------|---------|
| MLflow link       | [link]() — do not paste raw numbers here |
| Hypothesis result | ✅ Confirmed / ❌ Rejected / ⚠️ Partial |
| Observed patterns |         |
| Surprises         |         |
| Next experiment   | (specific action items) |
| Team decision     | ✅ Adopt / ⚠️ Conditional / ❌ Reject / ⏸️ Hold — [Decision Log](link) |

Failure analysis (fill in even for partial failures):
| Condition | Observed behaviour | Suspected cause |
|-----------|--------------------|-----------------|
```
