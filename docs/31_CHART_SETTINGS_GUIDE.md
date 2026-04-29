# Chart Settings Guide

MLflow stores runs-table column visibility in the **browser's localStorage**, which means settings are lost whenever you open a new browser or switch machines. This guide explains how to persist those settings so the same column layout is available to everyone on the team.

---

## How it works

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

## Persistence characteristics

| Scenario | Result |
|---|---|
| MLflow server restart | Settings survive — stored in the MLflow DB |
| Browser restart / new tab | Column layout lost — run the bookmarklet once to restore |
| Different team member opens the page | Same — run the bookmarklet once in their browser |
| Different browser or machine | Same — run the bookmarklet once |

The bookmarklet is a one-time action per browser. Once run, the settings persist until the browser's localStorage is cleared.

---

## Setup

### 1. Edit `chart_settings/chart_settings.json`

Define which columns and charts you want for each experiment:

```json
{
  "version": "1.0",
  "experiments": {
    "robot_hand_rl": {
      "visible_columns": {
        "tags":    ["researcher", "task", "seed", "hardware"],
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

### 2. Save settings to the MLflow server

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

### 3. Restore settings in the browser (bookmarklet)

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

## Verify stored settings

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

## Updating settings

1. Edit `chart_settings/chart_settings.json`.
2. Re-run `apply`.
3. Each team member runs the bookmarklet once in their browser to pick up the new layout.

---

## Reference

```
chart_settings/
├── chart_settings.json       — column and chart configuration (edit this)
└── apply_chart_settings.py   — CLI: apply / show / bookmarklet
```

```
python chart_settings/apply_chart_settings.py --help
```
