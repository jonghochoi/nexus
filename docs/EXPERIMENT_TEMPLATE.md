# Experiment Page Template

> Copy this template for every new experiment group in Confluence.
> Fill in the **Objective & Hypothesis section before training starts**.
> All other sections are filled in after results are available.

---

# [Experiment Name] — [Short Description]

**Space:** Robot Hand / Experiment Log  
**Owner:** @name  
**Date started:** YYYY-MM-DD  
**Status:** 🟡 In Progress / 🟢 Complete / 🔴 Abandoned

---

## 1. Objective & Hypothesis

**Problem being addressed:**
> Describe the specific failure mode, limitation, or open question
> this experiment is trying to resolve.

**Hypothesis:**
> State a specific, falsifiable prediction with a measurable threshold.
> Example: "Adding fingertip normal force to the reward signal will
> improve grasp_stability metric by >15% vs. baseline at 10M steps."

**Why this matters:**
> One sentence on how this connects to the broader project goal.

---

## 2. MLflow Runs

> Link to runs — do NOT copy metric values into this table.
> Open the MLflow link for all quantitative comparisons.

| Run Name | MLflow Link | Researcher | Status | Notes |
|---|---|---|---|---|
| `run_name_v1` | [link](http://mlflow-server:5000) | @name | Done | Baseline |
| `run_name_v2` | [link](http://mlflow-server:5000) | @name | Running | +tactile reward |

**Key MLflow tags used:**
```
seed=42
isaac_lab_version=x.x.x
physx_solver=TGS
hardware=robot_22dof
task=in_hand_reorientation
researcher=name
```

---

## 3. Key Findings

> Interpret what the results mean in plain language.
> Do not paste numbers — reference curves and trends only.
> If hypothesis was confirmed or refuted, state it clearly.

**Hypothesis outcome:** ✅ Confirmed / ❌ Refuted / ⚠️ Partial

**Summary:**
- [What the learning curves showed]
- [Which condition performed best and why]
- [Any unexpected behavior observed]

---

## 4. Failure Analysis / Edge Cases

> Document conditions where the policy broke down.
> This section is the most important one for the team's long-term learning.

| Condition | Observed Behavior | Suspected Cause |
|---|---|---|
| Object rotation > 30° | Grasp slip | Reward does not capture angular slip |
| High contact force | Reward collapse | Saturation in tactile sensor model |

**Sim artifacts suspected:**
> Note any behavior that may be a PhysX / Isaac Lab artifact rather than
> a genuine policy failure (e.g. contact normal direction mismatch).

---

## 5. Next Steps

> Concrete, actionable items only. Assign an owner to each.

- [ ] [Next experiment to run] — @owner
- [ ] [Parameter or reward weight to change] — @owner
- [ ] [Hypothesis to test in the next ablation] — @owner

---

## 6. Decision

**Outcome:** Adopted / Partially adopted / Rejected / Deferred  
**Date decided:** YYYY-MM-DD  
**Decided by:** Team / @name  

**Rationale:**
> One paragraph. Explain why the team made this decision based on the findings.
> Future team members should be able to read this and understand the reasoning
> without needing to re-read the full experiment.

---

*Link this page in the [Decision Log](../decision_log) if a team-level decision was made.*
