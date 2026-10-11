"""Audit saved AudioMNIST outer-fold results and make paired comparisons.

Read-only: never loads pickle, edits runs, or treats a full-fit model as a test.
Usage: python compare_results.py PATH_TO_RUNS_MAIN --out PATH_TO_OUTPUT_FOLDER
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

CONDITIONS = ("clean", "white_20", "white_10", "pink_10")
REPS = ("logmel", "mfcc_delta")
MODELS = ("logistic", "forest", "lightgbm_goss")
MODES = ("clean_matched", "augmented")


def load_run(root, rep, model, mode):
    path = root / f"{rep}__{model}__{mode}" / "result.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record["config"]
    assert (config["representation"], config["model"], config["mode"]) == (rep, model, mode)
    assert config["outer_folds"] == 10 and config["inner_folds"] == 3
    assert len(config["grid"]) == 3 and len(set(config["grid"])) == 3
    folds = {row["fold"]: row for row in record["folds"]}
    assert set(folds) == set(range(1, 11)), f"Incomplete folds: {path}"
    for row in folds.values():
        assert row["selected"] in config["grid"]
        assert len(row["inner_scores"]) == 3
        assert all(len(scores) == 3 for scores in row["inner_scores"])
        calculated = [mean(scores) for scores in row["inner_scores"]]
        assert config["grid"][max(range(3), key=lambda i: calculated[i])] == row["selected"]
        assert set(row["metrics"]) == set(CONDITIONS)
    return config, folds


def aligned(a, b):
    for fold in range(1, 11):
        x, y = a[fold], b[fold]
        assert x["test_people"] == y["test_people"]
        assert x["test_n"] == y["test_n"]
        assert x["inner_valid_people"] == y["inner_valid_people"]


def summarize(values):
    return {"mean": mean(values), "fold_sd": stdev(values),
            "positive_folds": sum(v > 0 for v in values),
            "negative_folds": sum(v < 0 for v in values),
            "zero_folds": sum(v == 0 for v in values)}


def analyse(root):
    runs, summary, pairs = {}, [], []
    for rep in REPS:
        for model in MODELS:
            for mode in MODES:
                config, folds = load_run(root, rep, model, mode)
                runs[rep, model, mode] = folds
                counts = [sum(row["selected"] == value for row in folds.values()) for value in config["grid"]]
                compliant = counts[1] > max(counts[0], counts[2])
                for condition in CONDITIONS:
                    scores = [folds[i]["metrics"][condition]["macro_f1"] for i in range(1, 11)]
                    summary.append({"representation": rep, "model": model, "mode": mode,
                                    "condition": condition, "macro_f1_mean": mean(scores),
                                    "macro_f1_fold_sd": stdev(scores), "grid": config["grid"],
                                    "selected_counts": counts, "middle_unique_mode": compliant,
                                    "report_role": "primary_eligible" if compliant else "exploratory_grid_failure"})
    for rep in REPS:
        for model in MODELS:
            clean, aug = runs[rep, model, "clean_matched"], runs[rep, model, "augmented"]
            aligned(clean, aug)
            clean_ok = next(row["middle_unique_mode"] for row in summary if
                            (row["representation"], row["model"], row["mode"]) == (rep, model, "clean_matched"))
            for condition in CONDITIONS:
                differences = [aug[i]["metrics"][condition]["macro_f1"] -
                               clean[i]["metrics"][condition]["macro_f1"] for i in range(1, 11)]
                pairs.append({"comparison": "augmentation_minus_clean_training", "representation": rep,
                              "model": model, "condition": condition, **summarize(differences),
                              "report_role": "primary_eligible" if clean_ok else "exploratory_grid_failure"})
    for model in MODELS:
        for mode in MODES:
            logmel, mfcc = runs["logmel", model, mode], runs["mfcc_delta", model, mode]
            aligned(logmel, mfcc)
            for condition in CONDITIONS:
                differences = [mfcc[i]["metrics"][condition]["macro_f1"] -
                               logmel[i]["metrics"][condition]["macro_f1"] for i in range(1, 11)]
                pairs.append({"comparison": "mfcc_delta_minus_logmel", "representation": "paired",
                              "model": model, "mode": mode, "condition": condition, **summarize(differences),
                              "report_role": "exploratory_grid_failure" if mode == "clean_matched" and model != "forest"
                              else "primary_eligible"})
    return summary, pairs


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})


def write_outputs(out, summary, pairs):
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "configuration_summary.csv", summary)
    write_csv(out / "paired_differences.csv", pairs)
    # Core is fixed by the experiment design and audited grid status; it is not
    # selected by whichever outer-test score happens to be highest.
    core = [row for row in summary if row["representation"] == "logmel" or
            (row["representation"] == "mfcc_delta" and row["model"] == "forest")]
    assert all(row["middle_unique_mode"] for row in core)
    write_csv(out / "assignment_core.csv", core)
    (out / "comparison.json").write_text(json.dumps({"summary": summary, "pairs": pairs},
                                                ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_main", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, pairs = analyse(args.runs_main)
    write_outputs(args.out, summary, pairs)
    failures = sorted({(r["representation"], r["model"], r["mode"]) for r in summary
                       if not r["middle_unique_mode"]})
    print(f"AUDIT_COMPLETE: {len(summary)//4} configurations, {len(pairs)} paired comparisons")
    print("GRID_FAILURES:", failures)


if __name__ == "__main__":
    main()
