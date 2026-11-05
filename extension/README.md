# Stress Estimator — Chrome extension

Real-time stress feedback while you type anywhere in Chrome: Gmail, Google
Docs, chat apps, forms. Everything runs inside your browser. There is no server,
and nothing you type leaves your computer.

## Install

1. In Chrome open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select this `extension/` folder.
4. Pin it: puzzle-piece icon in the toolbar → pin **Stress Estimator**.

Works in any Chromium browser (Chrome, Edge, Brave). Python is not needed.

## Use

Type normally in any text box. After you pause for 2 seconds (and have typed
at least 15 keys), the badge on the toolbar icon updates:

| Badge | Meaning |
|---|---|
| `LOW` / `MED` / `HIGH` | average of your last few typing bursts (last 10 minutes) |
| no badge | nothing typed in the last 10 minutes |
| `OFF` | extension switched off in the popup |

Click the icon for details: the recent level with probabilities, what the
typing model and the text model each said about the last burst, a colour strip
of recent estimates, and the controls:

- **On / off** switch
- **Pause on this site**, e.g. for a banking or work site
- **Suggest a break** — a notification when the last 3 bursts were all High
  (at most once every 30 minutes)

## How far to trust it

- The reading comes mostly from your words. The text model scores macro F1 0.761 on Dreaddit's
  official test set, which is Reddit writing, so everyday typing may differ.
- Typing on a single burst is at chance on people the model has not seen. It gets a weight of
  0.25, shows as "Unclear" below 0.5 confidence, and an unclear burst with too few words is
  skipped entirely.
- The text model only knows Low and High, so a burst with enough words is never labelled Medium.
  `MED` is therefore rare, and appears only when recent bursts were mostly typing with few words.

It is for self-reflection, not diagnosis.

## See your stress over time

Click **View history** in the popup for the dashboard:

- **Period:** Today, Yesterday, 7 days, 30 days, or any dates you pick.
- **Overall:** your average stress score (0 to 100), a plain verdict, the share of Low, Medium
  and High, and the change from the previous period.
- **What stands out:** short sentences, only for differences of 10+ points on 3+ estimates.
- **Through the day:** every estimate on an axis zoomed to when you typed, with a rolling
  average. Longer periods show the average per day.
- **By site:** sites ranked by score. Click one to filter the whole page.
- **By day and time:** a weekday and time-of-day heatmap, for periods longer than a day.
- **Table view:** the same numbers as tables.

Periods and sites with fewer than 3 estimates are drawn faded and are never called out,
because so few estimates say little. If you just updated the extension, press the reload
button on its card in `chrome://extensions` first.

## What it does and does not read

- Reads key-press and key-release timings and the characters typed **in text
  fields** (`<textarea>`, text/search inputs, rich-text editors).
- Never reads password fields, email fields, credit-card or one-time-code
  fields.
- Analyses only what was typed during the burst, in memory, inside the browser. It makes no
  network requests and has no host permissions.
- Stores only *estimates* (level, probabilities, site name, time) in Chrome's extension
  storage, for 30 days at most. Never the typed text. **Clear** in the popup or **Delete
  all history** on the dashboard removes them.

## How it fits the project

```
any web page ──content.js──► background.js ──► engine.js + model.js
 (typing)      one burst      badge, history,    (features and models, matching Python)
                              break suggestion
```

`python export_extension_model.py` writes `model.js`: the text model from Dreaddit and the typing
model from SWELL-KW bursts. `tests/test_engine_parity.py` checks that the JavaScript gives the same
features and probabilities as Python. Retraining with `train_demo.py` changes only the demo web
app. To change the extension's models, edit `fit_models` in the export script and run it again.

## Files

| File | Role |
|---|---|
| `manifest.json` | Manifest V3; permissions: storage, alarms, notifications, activeTab; no host permissions |
| `content.js` | runs in pages; captures typing bursts in text fields |
| `background.js` | runs the engine, smooths estimates, badge, notifications |
| `engine.js` | sentiment, features and the two models in JavaScript |
| `recent.js` | the recent level, shared by the badge and popup; estimates expire after 10 minutes |
| `model.js` | the trained model weights (generated) |
| `popup.html/css/js` | the panel shown when the icon is clicked |
| `dashboard.html/css/js`, `dashboard-data.js` | the history page: periods, summary, charts, table |
| `icons/` | toolbar icons |
