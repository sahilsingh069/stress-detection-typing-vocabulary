# Data Collection Protocol

How to run one participant session with `collector/typing_collector.html` (v2,
chat interview). Follow it the same way every time. Small differences between
sessions (a noisy room, a different keyboard, someone watching) add noise that
the model will try to learn.

**Target:** 20–30 participants. Each yields 10 samples (2 copy tasks + 8 chat
answers) in about 10–15 minutes.

---

## Before collecting at all

- [ ] **Ethics clearance resolved** with Dr. Souvik Dev (Mentor Agenda item A4).
      Do not collect until you know whether a signed consent form is required.
- [ ] Participant codes decided: `P01`, `P02`, ... Never names, roll numbers or
      emails. If you keep a sign-up list with names, keep it on paper or in a
      separate file, never next to the data.
- [ ] One laptop with a **physical keyboard** for all sessions if possible.
      Phones and tablets are not supported: on-screen keyboards do not report
      reliable key timings.
- [ ] Google Chrome (or Edge). Open the collector once and click through to
      check it loads.

## Setting up a session

1. Quiet room, participant alone at the laptop. You stay nearby but out of
   sight of the screen.
2. Close other apps and notifications.
3. Open `collector/typing_collector.html` (double-click; works offline).
4. If a red-dashed box says a recording from an earlier visit is saved,
   either download it (if that participant's files were never saved) or
   discard it.
5. Enter the participant code, leave age and typing familiarity for the
   participant to choose.

## What to say (read it, same words every time)

> "This takes about fifteen minutes. You'll copy a short passage and answer a
> few everyday questions in a chat, first at your own pace and then against a
> timer. After each part you rate how stressed you felt. There are no right
> answers and nothing is scored. Please don't type any personal details like
> your name or phone number. Everything stays on this laptop. You can stop at
> any time."

Then let them read the consent screen and tick the box themselves.

## During the session

- Do not help, comment, or watch the screen. Encouragement or impatience
  changes how people type.
- Do not explain the purpose beyond the script ("we're studying stress"):
  people who know what is measured change their behaviour.
- If they ask whether they're "doing it right": "Whatever feels natural is
  right."
- Pasting is blocked on purpose.

## After the session

1. On the final screen click **Download both CSV files**. You should get
   `sessions_P07_SXXXXX.csv` and `keystrokes_P07_SXXXXX.csv`.
2. Move both into `data/own/` in the project.
3. Check they load:

   ```bash
   python -c "from dataio import loaders; loaders.load_own_paired()"
   ```

   It should report 10 sessions for the new participant (a timed-out answer
   with very few keys may be dropped from typing; that is expected).
4. Click **Record another participant** only after the files are safely in
   `data/own/`. It clears the browser's copy.
5. Note anything unusual (interruption, participant unwell, wrong code) in a
   session log next to the code, never next to a name. Drop a session from
   the analysis by moving its two files out of `data/own/`, not by editing them.

## If something goes wrong

| Problem | Do this |
|---|---|
| Tab closed or browser crashed mid-session | Reopen the collector; the recovery box offers the rated blocks so far as CSV + JSON. Unrated answers are only in the JSON. |
| Participant wants to stop | Let them. Discard their recovery data unless they agree to keep the completed blocks. |
| Wrong participant code typed | Finish the session, then rename both files and fix the `participant_id` column consistently in both. |
| Same person comes back | Do not record them twice. Repeating the questions defeats the randomisation. |

## Data handling

- `data/own/` is git-ignored. Never commit, email or upload participant files.
- Keep files only on team laptops; delete them at the end of the project
  unless the department requires retention.
- Report results only in aggregate. Never quote a participant's typed text
  in the report or slides without their permission.

## When collection is done

```bash
python run_final.py
```

Read `outputs/paired/report_paired.md`. **Check the manipulation check
first.** If the timed blocks were not rated as more stressful, the study did
not induce stress, and that has to be stated before any model result. Then
`python train_demo.py` so the demo uses models trained on your participants.
