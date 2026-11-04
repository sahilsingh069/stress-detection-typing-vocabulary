# Stress Estimator — Chrome extension

Real-time stress feedback while you type anywhere in Chrome: Gmail, Google
Docs, chat apps, forms. The extension measures your typing, sends it to the
project's server **on your own computer**, and shows the result on the
toolbar.

## Install (once)

1. In the project folder, train the models and start the server:

   ```bash
   python train_demo.py
   python app/server.py          # keep this terminal open
   ```

2. In Chrome open `chrome://extensions`.
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and select this `extension/` folder.
5. Pin it: puzzle-piece icon in the toolbar → pin **Stress Estimator**.

Works in any Chromium browser (Chrome, Edge, Brave).

## Use

Type normally in any text box. After you pause for 2 seconds (and have typed
at least 15 keys), the badge on the toolbar icon updates:

| Badge | Meaning |
|---|---|
| `LOW` / `MED` / `HIGH` | average of your last few typing bursts (last 10 minutes) |
| `!` | server not running: start `python app/server.py` |
| `OFF` | extension switched off in the popup |

Click the icon for details: the recent level with probabilities, what the
typing model and the text model each said about the last burst, a colour strip
of recent estimates, and the controls:

- **On / off** switch
- **Pause on this site**, e.g. for a banking or work site
- **Suggest a break** — a notification when the last 3 bursts were all High
  (at most once every 30 minutes)

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
- Sends only what was typed during the burst — not the rest of the page or
  the field — and only to `http://127.0.0.1:8765` (this computer). The server
  analyses it in memory and keeps nothing.
- Stores only *estimates* (level, probabilities, site name, time) in Chrome's extension
  storage, for 30 days at most. Never the typed text. **Clear** in the popup or **Delete
  all history** on the dashboard removes them.

## How it fits the project

```
any web page ──content.js──► background.js ──POST /api/predict──► app/server.py
 (typing)      one burst      badge, history,     (same feature code and models
                              break suggestion      as the experiments)
```

The models are whatever `train_demo.py` built: until own data is collected,
the typing model comes from the SWELL-KW raw logs (23 people, 5-minute windows)
and the popup says "unvalidated". After collecting data, re-run `python train_demo.py` and
restart the server; the extension picks up the new models automatically.

## Files

| File | Role |
|---|---|
| `manifest.json` | Manifest V3; permissions: storage, notifications, activeTab; network access only to `127.0.0.1:8765` |
| `content.js` | runs in pages; captures typing bursts in text fields |
| `background.js` | calls the local server, smooths estimates, badge, notifications |
| `popup.html/css/js` | the panel shown when the icon is clicked |
| `dashboard.html/css/js`, `dashboard-data.js` | the history page: periods, summary, charts, table |
| `icons/` | toolbar icons |
