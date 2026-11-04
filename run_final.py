"""
Final run - every experiment the available data supports, one command.

    python run_final.py               # full run (grid search: a few minutes)
    python run_final.py --quick       # skip the Kaggle/Dreaddit grid search and the
                                      # EmoSurv permutation tests, reuse their results

Runs, in order, whatever has data in place:

    1. Text pipeline on Dreaddit and typing pipeline on Kaggle (run_midterm)
    2. SWELL-KW typing experiment                   (run_swell)   if data/swell_kw.csv
    3. Paired experiments: SWELL-KW raw logs        (run_swell_paired) if downloaded
       and own collected sessions                   (run_paired)  if data/own/
    4. EmoSurv typing, text and fusion              (run_emosurv) if its files are in data/
    5. Transformer comparison results, if run_transformer.py has been run
    6. Demo models for the web app                  (train_demo)

and writes outputs/final_results.csv plus outputs/final_report.md, which
states for every component whether it ran and what it found.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

import config
from dataio import emosurv, loaders

warnings.filterwarnings("ignore")
OUT = config.OUTPUT_DIR


def _rel(path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _best(df: pd.DataFrame, **filters) -> pd.Series | None:
    for k, v in filters.items():
        df = df[df[k] == v]
    return None if df.empty else df.loc[df["f1_macro"].idxmax()]


def main(quick: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    status, frames = {}, []

    # 1. mid-term pipelines --------------------------------------------------
    midterm_csv = OUT / "midterm_results.csv"
    if quick and midterm_csv.exists():
        print("--quick: reusing", midterm_csv)
        status["midterm"] = "reused previous run"
    else:
        import run_midterm
        run_midterm.main()
        status["midterm"] = "ran"
    mid = pd.read_csv(midterm_csv)
    mid["dataset"] = mid["pipeline"].map({"text": "Dreaddit", "typing": "Kaggle keystroke"})
    mid["evaluation"] = "random 80/20 split"
    frames.append(mid)

    # 2. SWELL-KW -------------------------------------------------------------
    if config.SWELL_PATH.exists():
        import run_swell
        sw = run_swell.main()
        sw = sw.assign(dataset="SWELL-KW", pipeline="typing",
                       variant=sw["variant"] + " / " + sw["features"],
                       evaluation=sw["evaluation"].replace(
                           {"grouped": "participants held out",
                            "calibrated": "participants held out, calibrated"}))
        frames.append(sw)
        status["swell"] = "ran"
    else:
        status["swell"] = f"skipped: no file at {_rel(config.SWELL_PATH)}"

    # 3. paired data ----------------------------------------------------------
    from dataio import swell_raw
    if swell_raw.ULOG_DIR.exists() and any(swell_raw.ULOG_DIR.glob("*.xml")):
        import run_swell_paired
        sp = run_swell_paired.main()
        frames.append(sp.assign(dataset="SWELL-KW paired", pipeline="paired",
                                variant=sp["task"] + " / " + sp["mode"],
                                evaluation="participants held out"))
        status["swell_paired"] = "ran"
    else:
        status["swell_paired"] = "skipped: run python -m dataio.swell_raw to download the logs"

    if loaders.own_data_available():
        import run_paired
        pr = run_paired.main("own")
        pr = pr.assign(dataset="Own collected (paired)", pipeline="paired",
                       variant=pr["mode"], evaluation="participants held out")
        frames.append(pr)
        status["paired"] = "ran"
    else:
        status["paired"] = f"skipped: no sessions_*.csv + keystrokes_*.csv in {_rel(config.OWN_DIR)}/"

    # 4. EmoSurv --------------------------------------------------------------
    emo_csv = OUT / "emosurv" / "typing_free.csv"
    if quick and emo_csv.exists():
        status["emosurv"] = "reused previous run"
    elif emosurv.available():
        import run_emosurv
        run_emosurv.main()
        status["emosurv"] = "ran"
    else:
        status["emosurv"] = "skipped: EmoSurv files not in data/ (see data/README.md)"
    if emo_csv.exists():
        em = pd.read_csv(emo_csv)
        frames.append(pd.DataFrame({
            "dataset": "EmoSurv (emotion labels)", "pipeline": "typing", "variant": em["task"] + " / free text",
            "evaluation": "participants held out", "model": "LogisticRegression, own baseline",
            "accuracy": em["accuracy"], "f1_macro": em["primary_f1"], "majority_baseline_acc": em["majority_acc"]}))

    # 5. transformer ----------------------------------------------------------
    tf_csv = OUT / "transformer" / "results_transformer.csv"
    if tf_csv.exists():
        tf = pd.read_csv(tf_csv)
        if not tf["subset"].any():
            frames.append(tf.assign(dataset="Dreaddit", evaluation="random 80/20 split"))
            status["transformer"] = "included (from run_transformer.py)"
        else:
            status["transformer"] = "skipped: only a subset smoke test exists"
    else:
        status["transformer"] = "not run (optional: python run_transformer.py)"

    # 6. demo models ----------------------------------------------------------
    import train_demo
    meta = train_demo.main()
    status["demo"] = ", ".join(f"{k}: {v['source']}" for k, v in meta["models"].items())

    # combine -----------------------------------------------------------------
    cols = ["dataset", "pipeline", "variant", "evaluation", "model", "accuracy",
            "precision_macro", "recall_macro", "f1_macro", "majority_baseline_acc"]
    allres = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    allres = allres.round(4)
    allres.to_csv(OUT / "final_results.csv", index=False)
    write_report(allres, status)

    print("\n================ FINAL STATUS ================")
    for k, v in status.items():
        print(f"  {k:12s} {v}")
    print(f"\nwritten: {OUT / 'final_results.csv'}, {OUT / 'final_report.md'}")


def write_report(res: pd.DataFrame, status: dict):
    lines = ["# Final Results", "",
             "Stress Level Detection Using Typing Behaviour and Text Vocabulary", "",
             "## What ran", ""]
    lines += [f"- **{k}**: {v}" for k, v in status.items()]
    lines += ["", "## Best result per dataset and pipeline", ""]
    # Random splits that mix people across folds are not eligible: they inflate scores.
    best = (res[~res["evaluation"].str.contains("people mixed")].sort_values("f1_macro", ascending=False)
               .groupby(["dataset", "pipeline"], as_index=False).head(1))
    lines += [best.to_markdown(index=False), ""]

    lines += ["## Findings", ""]
    t = _best(res, dataset="Dreaddit", pipeline="text", variant="tfidf+linguistic")
    if t is not None:
        lines.append(f"- Text vocabulary works: {t['model']} reaches macro F1 "
                     f"{t['f1_macro']:.3f} against a {t['majority_baseline_acc']:.3f} "
                     "majority baseline on Dreaddit.")
    tf = _best(res, dataset="Dreaddit", variant="fine-tuned transformer")
    if t is not None and tf is not None:
        lines.append(f"- A fine-tuned transformer ({tf['model']}) reaches macro F1 "
                     f"{tf['f1_macro']:.3f} on the same split ({tf['f1_macro'] - t['f1_macro']:+.3f}): "
                     "the interpretable classical pipeline gives up only that much.")
    k = _best(res, dataset="Kaggle keystroke")
    if k is not None:
        lines.append(f"- The 2-person Kaggle keystroke set stays below its baseline "
                     f"(best macro F1 {k['f1_macro']:.3f}, accuracy {k['accuracy']:.3f} vs "
                     f"{k['majority_baseline_acc']:.3f}); its cross-user test shows why.")
    s = res[res.dataset == "SWELL-KW"]
    if not s.empty:
        held = s[s.evaluation == "participants held out"]
        g = _best(held, variant="3class / all interaction")
        nt = _best(held, variant="n_vs_t / all interaction")
        kb = _best(held, variant="n_vs_t / keyboard only")
        if g is not None:
            lines.append(f"- SWELL-KW (25 people, new participants held out), 3-class: "
                         f"macro F1 {g['f1_macro']:.3f} vs 0.333 chance; accuracy "
                         f"{g['accuracy']:.3f} vs {g['majority_baseline_acc']:.3f} majority.")
        if nt is not None:
            lines.append(f"- SWELL-KW neutral vs. time pressure (neutral is always the "
                         f"first block): best-of-three macro F1 {nt['f1_macro']:.3f} vs 0.500 chance"
                         + (f", {kb['f1_macro']:.3f} from keyboard features alone" if kb is not None else "")
                         + f"; accuracy {nt['accuracy']:.3f} vs {nt['majority_baseline_acc']:.3f} "
                         "majority. A weak signal on unseen people; see "
                         "`evaluation/report_evaluation.md` for significance.")
    sp = res[res.dataset == "SWELL-KW paired"]
    if not sp.empty:
        lines.append("- SWELL-KW paired (typing and text from the same windows): see "
                     "`paired_swell/report_paired_swell.md` and, for significance and "
                     "confidence intervals, `evaluation/report_evaluation.md`.")
    ens = OUT / "transformer" / "ensemble_official.csv"
    if ens.exists():
        e = pd.read_csv(ens).set_index("model").loc["Ensemble (declared headline)"]
        lines.append(f"- Dreaddit official split: an ensemble declared in advance reaches macro F1 "
                     f"{e['macro_f1']:.3f} [{e['f1_lo']:.3f}, {e['f1_hi']:.3f}]; see "
                     "`transformer/report_ensemble_official.md`.")
    emo = OUT / "emosurv" / "report_emosurv.md"
    if emo.exists():
        lines.append("- EmoSurv (emotion labels, about 80 people held out): see `emosurv/report_emosurv.md`. "
                     "Why published scores are higher: `leakage/report_leakage.md`.")
    p = res[res.dataset == "Own collected (paired)"]
    if not p.empty:
        by = p.groupby("variant")["f1_macro"].max()
        single = max(by.get("typing_only", 0), by.get("text_only", 0))
        fused = by.drop(labels=["typing_only", "text_only", "typing_only_all_tasks"],
                        errors="ignore")
        if len(fused):
            lines.append(f"- Own paired data: best single modality macro F1 "
                         f"{single:.3f}; best fusion {fused.max():.3f} "
                         f"({fused.idxmax()}), a change of {fused.max() - single:+.3f}.")
    else:
        lines.append("- Own paired data: not collected yet. To add it, run the collector with "
                     "participants, put the "
                     "exports in data/own/, and re-run this script.")
    lines += ["", "## All results", "", res.to_markdown(index=False), "",
              "Detailed reports: `evaluation/report_evaluation.md` (from "
              "`run_evaluation.py`, run separately), `midterm_report.md`, `swell/report_swell.md`, "
              "`paired/report_paired.md`, `transformer/report_transformer.md`.", ""]
    (OUT / "final_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reuse outputs/midterm_results.csv instead of re-running it")
    main(ap.parse_args().quick)
