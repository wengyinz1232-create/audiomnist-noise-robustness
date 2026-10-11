"""Compare two completed training modes on identical speaker folds.

Usage: python compare_training_modes.py OLD_RESULT_JSON NEW_RESULT_JSON --out FILE.json
Positive delta means the new mode increased macro-F1. No pickle is loaded.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

CONDITIONS = ("clean", "white_20", "white_10", "pink_10")


def read(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = {row["fold"]: row for row in doc["folds"]}
    assert set(folds) == set(range(1, 11)), f"Need ten complete outer folds: {path}"
    assert doc["config"]["inner_folds"] == 3
    return doc["config"], folds


def compare(old_path, new_path):
    old_config, old = read(old_path)
    new_config, new = read(new_path)
    for key in ("representation", "samples", "features", "seed", "grid"):
        assert old_config[key] == new_config[key], f"Mismatched {key}"
    assert old_config["model"] == new_config["model"]
    for fold in range(1, 11):
        for key in ("train_people", "test_people", "inner_valid_people", "train_n", "test_n"):
            assert old[fold][key] == new[fold][key], f"Mismatched {key}, fold {fold}"
    selection = {}
    for label, config, rows in (("old", old_config, old), ("new", new_config, new)):
        counts = [sum(row["selected"] == value for row in rows.values()) for value in config["grid"]]
        selection[label] = {"grid": config["grid"], "counts": counts,
                            "middle_unique_mode": counts[1] > max(counts[0], counts[2])}
    metrics = {}
    for condition in CONDITIONS:
        old_values = [old[i]["metrics"][condition]["macro_f1"] for i in range(1, 11)]
        new_values = [new[i]["metrics"][condition]["macro_f1"] for i in range(1, 11)]
        differences = [b-a for a,b in zip(old_values,new_values)]
        metrics[condition] = {"old_mean": mean(old_values), "new_mean": mean(new_values),
                              "new_minus_old_mean": mean(differences),
                              "paired_fold_sd": stdev(differences),
                              "positive_folds": sum(x > 0 for x in differences),
                              "negative_folds": sum(x < 0 for x in differences),
                              "zero_folds": sum(x == 0 for x in differences)}
    return {"representation": old_config["representation"], "model": old_config["model"],
            "old_mode": old_config["mode"], "new_mode": new_config["mode"],
            "selection": selection, "metrics": metrics,
            "note": "Descriptive paired fold differences; folds overlap and this is not an independent confirmation set."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PAIRED_COMPARISON_COMPLETE", args.out)


if __name__ == "__main__":
    main()
