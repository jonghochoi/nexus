# 📊 Chart Settings

> **Purpose:** Persist MLflow runs-table column visibility (which tags, params, and metrics are shown) as **experiment tags on the server**, so the same column layout survives browser restarts and is shared across the whole team.
>
> **Audience:** Operators / leads who curate the team-standard MLflow UI.

---

## 📑 Table of Contents

- [🧩 1. How it works](#-1-how-it-works)
- [🧩 2. Persistence characteristics](#-2-persistence-characteristics)
- [🧩 3. Setup](#-3-setup)
- [🧩 4. Verify stored settings](#-4-verify-stored-settings)
- [🧩 5. Updating settings](#-5-updating-settings)
- [🧩 6. Reference](#-6-reference)
- [🗺️ Next steps](#-next-steps)

---

## 🧩 1. How it works

Settings are stored as **MLflow experiment tags** on the server. Because tags live in the MLflow database they survive server restarts and are shared across all team members. When you open a fresh browser, a one-click bookmarklet reads those tags and restores the column layout in localStorage.

```
chart_settings.json  ──apply──►  MLflow experiment tags  (server, permanent)
                                          │
                                    bookmarklet
                                          │
                                          ▼
                                 browser localStorage  (per browser session)
```

---

## 🧩 2. Persistence characteristics

| Scenario | Result |
|---|---|
| MLflow server restart | Settings survive — stored in the MLflow DB |
| Browser restart / new tab | Column layout lost — run the bookmarklet once to restore |
| Different team member opens the page | Same — run the bookmarklet once in their browser |
| Different browser or machine | Same — run the bookmarklet once |

The bookmarklet is a one-time action per browser. Once run, the settings persist until the browser's localStorage is cleared.

---

## 🧩 3. Setup

### 3.1 Edit `chart_settings/chart_settings.json`

Define which columns and charts you want for each experiment:

```json
{
  "version": "1.0",
  "experiments": {
    "robot_hand_rl": {
      "visible_columns": {
        "tags":    ["experiment", "researcher", "task", "hardware"],
        "params":  [],
        "metrics": ["train/reward_mean", "eval/reward_mean", "rl/success_rate"]
      },
      "charts": [
        {
          "title": "Training Reward",
          "type": "line",
          "metrics": ["train/reward_mean"],
          "x_axis": "step",
          "group_by_tag": "researcher"
        }
      ]
    }
  }
}
```

`visible_columns` controls which columns appear in the runs table. Add or remove entries freely — they must match the tag keys, param keys, and metric names your runs actually log.

### 3.2 Save settings to the MLflow server

```bash
python chart_settings/apply_chart_settings.py apply
```

To target a single experiment:

```bash
python chart_settings/apply_chart_settings.py apply --experiment real_robot_eval
```

To point at a non-default server:

```bash
python chart_settings/apply_chart_settings.py apply --tracking-uri http://nexus-server:5000
```

Run this once after editing `chart_settings.json`. Re-run whenever you update the file.

### 3.3 Restore settings in the browser (bookmarklet)

Generate the bookmarklet JavaScript:

```bash
python chart_settings/apply_chart_settings.py bookmarklet
```

**Option A — browser console (quickest):**
1. Open the MLflow page and press `F12` to open DevTools.
2. Go to the **Console** tab.
3. Paste the printed JavaScript and press Enter.
4. The page reloads with your column layout applied.

**Option B — saved bookmark (recommended for regular use):**
1. Create a new bookmark in your browser.
2. Set the URL field to the entire printed JavaScript (starting with `javascript:`).
3. Save it to your bookmarks bar.
4. Whenever you open MLflow in a fresh browser, click the bookmark — done.

This works the same way in Chrome, Firefox, Edge, and Safari. The JavaScript itself is identical across all browsers.

---

## 🧩 4. Verify stored settings

```bash
python chart_settings/apply_chart_settings.py show
```

Sample output:

```
MLflow server: http://nexus-server:5000

  experiment: robot_hand_rl  (v1.0)
    tag columns   : researcher, task, seed, hardware
    params        : (none)
    metric columns: train/reward_mean, eval/reward_mean, rl/success_rate
    chart 1: Training Reward — train/reward_mean
    chart 2: Eval Reward — eval/reward_mean
```

---

## 🧩 5. Updating settings

1. Edit `chart_settings/chart_settings.json`.
2. Re-run `apply`.
3. Each team member runs the bookmarklet once in their browser to pick up the new layout.

---

## 🧩 6. Reference

```
chart_settings/
├── chart_settings.json       — column and chart configuration (edit this)
└── apply_chart_settings.py   — CLI: apply / show / bookmarklet
```

```
python chart_settings/apply_chart_settings.py --help
```

---

## 🗺️ Next steps

- **Other opt-in features (SweepLogger, RL metrics, Model Registry)** → [`30_ADVANCED_FEATURES.md`](30_ADVANCED_FEATURES.md)
- **Architecture detail (where experiment tags fit in the data flow)** → [`10_ARCHITECTURE.md`](10_ARCHITECTURE.md)
