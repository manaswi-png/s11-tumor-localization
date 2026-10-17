"""
Phase 3 - Train and evaluate tumor-location classifiers with phantom-grouped validation.

Run from the project root (after 01_extract_clean.py):
    python scripts/02_train_evaluate.py

Methods compared
  A  rf_baseline   : the notebook's Random Forest on 504 band features (no changes)
  B  rf_rotaug     : same Random Forest, training set augmented with 90/180/270-degree
                     antenna-axis rotations (physics-based; training data only)
  C  physics_peak  : no ML - the antenna with the strongest |dS11| points at the tumor

Evaluation protocol (no phantom ever appears in both training and evaluation data)
  1. Validation : leave-one-phantom-out (LOPO) CV inside the 15 training phantoms -> choose method
  2. Test       : fit on all 15 training phantoms, evaluate ONCE on the 5 held-out test phantoms
  3. Extra      : LOPO over all 20 phantoms (more stable estimate, every phantom tested once)
  4. Leakage demo: the notebook's random stratified 5-fold CV, to show why it is optimistic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s11ml import core  # noqa: E402
from s11ml.methods import METHODS, Method  # noqa: E402

SEED = 42
LABELS = [0, 1, 2, 3]
PROC, MODELS, REPORTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
FIG = REPORTS / "figures"
for p in (MODELS, FIG):
    p.mkdir(parents=True, exist_ok=True)


def metrics(y_true, y_pred) -> dict:
    present = sorted(set(np.asarray(y_true).tolist()))   # macro-average over classes that exist in y_true
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    err = core.localization_error_cm(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "mean_localization_error_cm": float(err.mean()),
        "max_localization_error_cm": float(err.max()),
        "per_class": {core.CLASS_SHORT[c]: {"precision": float(pc[i]), "recall": float(rc[i]),
                                             "f1": float(fc[i]), "support": int(sc[i])}
                      for i, c in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def lopo(kind, X, y, groups):
    pred = np.empty_like(y)
    proba = np.zeros((len(y), 4))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = Method(kind).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        proba[te] = m.predict_proba(X[te])
    return pred, proba


def plot_cm(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=130)
    ax.imshow(cm, cmap="Blues")
    names = [core.CLASS_SHORT[c] for c in LABELS]
    ax.set_xticks(range(4), names); ax.set_yticks(range(4), names)
    ax.set_xlabel("Predicted location (x,y sign)"); ax.set_ylabel("True location")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    df = pd.read_parquet(PROC / "s11_features.parquet")
    fnames = core.feature_names()
    X, y = df[fnames].to_numpy(), df["label"].to_numpy()
    groups, diam = df["phant_id"].to_numpy(), df["tum_diam_cm"].to_numpy()
    tr, te = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    print(f"Train: {tr.sum()} samples / {len(set(groups[tr]))} phantoms | "
          f"Test: {te.sum()} samples / {len(set(groups[te]))} phantoms")
    print("Train class counts:", np.bincount(y[tr], minlength=4), " Test:", np.bincount(y[te], minlength=4))

    results = {"protocol": {}, "validation_lopo_train": {}, "test": {}, "lopo_all_20": {},
               "leaky_random_cv_notebook": {}}
    results["protocol"] = {
        "train_phantoms": sorted(set(groups[tr])), "test_phantoms": sorted(set(groups[te])),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "train_class_counts": {core.CLASS_SHORT[c]: int((y[tr] == c).sum()) for c in LABELS},
        "test_class_counts": {core.CLASS_SHORT[c]: int((y[te] == c).sum()) for c in LABELS},
        "task": "4-class classification of tumor position (x,y); z fixed at -6.5 cm",
    }
    pred_rows = []

    # 1. validation: LOPO inside training phantoms
    for k in METHODS:
        p, _ = lopo(k, X[tr], y[tr], groups[tr])
        results["validation_lopo_train"][k] = metrics(y[tr], p)
    # The deployed model is chosen among the two ML methods (decided before seeing test data).
    # The physics rule is a reference baseline and an independent cross-check in the website.
    best = max(["rf_baseline", "rf_rotaug"], key=lambda k: (results["validation_lopo_train"][k]["macro_f1"],
                                                            results["validation_lopo_train"][k]["accuracy"]))
    results["selected_method"] = best
    print("\nValidation (LOPO on 15 training phantoms):")
    for k in METHODS:
        v = results["validation_lopo_train"][k]
        print(f"  {k:13s} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} err={v['mean_localization_error_cm']:.2f} cm")
    print("Selected:", best)

    # 2. held-out test (fit on all training phantoms, evaluate once)
    fitted = {}
    for k in METHODS:
        m = Method(k).fit(X[tr], y[tr]); fitted[k] = m
        p, P = m.predict(X[te]), m.predict_proba(X[te])
        results["test"][k] = metrics(y[te], p)
        for i, idx in enumerate(np.where(te)[0]):
            pred_rows.append({"evaluation": "test", "method": k, "sample_id": int(df.sample_id[idx]),
                              "phant_id": groups[idx], "tum_diam_cm": diam[idx], "true": int(y[idx]),
                              "pred": int(p[i]), "confidence": float(P[i].max()),
                              **{f"p_{core.CLASS_SHORT[c]}": float(P[i, c]) for c in LABELS}})
        plot_cm(results["test"][k]["confusion_matrix"], f"Test set - {METHODS[k]}", FIG / f"cm_test_{k}.png")
    print("\nHeld-out TEST (5 unseen phantoms):")
    for k in METHODS:
        v = results["test"][k]
        print(f"  {k:13s} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} err={v['mean_localization_error_cm']:.2f} cm")

    # 3. LOPO over all 20 phantoms
    for k in METHODS:
        p, P = lopo(k, X, y, groups)
        r = metrics(y, p)
        r["accuracy_by_diameter"] = {f"{d:.1f}": float((p[diam == d] == y[diam == d]).mean())
                                     for d in sorted(set(diam))}
        r["accuracy_by_phantom"] = {g: float((p[groups == g] == y[groups == g]).mean()) for g in sorted(set(groups))}
        results["lopo_all_20"][k] = r
        for idx in range(len(y)):
            pred_rows.append({"evaluation": "lopo_all_20", "method": k, "sample_id": int(df.sample_id[idx]),
                              "phant_id": groups[idx], "tum_diam_cm": diam[idx], "true": int(y[idx]),
                              "pred": int(p[idx]), "confidence": float(P[idx].max()),
                              **{f"p_{core.CLASS_SHORT[c]}": float(P[idx, c]) for c in LABELS}})
        plot_cm(r["confusion_matrix"], f"LOPO, all 20 phantoms - {METHODS[k]}", FIG / f"cm_lopo_{k}.png")
    print("\nLOPO over all 20 phantoms:")
    for k in METHODS:
        v = results["lopo_all_20"][k]
        print(f"  {k:13s} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} err={v['mean_localization_error_cm']:.2f} cm "
              f"by diam={v['accuracy_by_diameter']}")

    # 4. the notebook's random CV (leaky) for comparison
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p = np.empty_like(y)
    for a, b in skf.split(X, y):
        p[b] = Method("rf_baseline").fit(X[a], y[a]).predict(X[b])
    results["leaky_random_cv_notebook"]["rf_baseline"] = metrics(y, p)
    print(f"\nNotebook-style random 5-fold CV (phantoms leak): acc={results['leaky_random_cv_notebook']['rf_baseline']['accuracy']:.3f}")

    # ---- figures: accuracy by diameter, feature importance ----
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    ds = sorted(set(diam)); w = 0.27
    for i, k in enumerate(METHODS):
        acc = [results["lopo_all_20"][k]["accuracy_by_diameter"][f"{d:.1f}"] for d in ds]
        ax.bar(np.arange(len(ds)) + (i - 1) * w, acc, w, label=METHODS[k].split(". ")[1])
    ax.set_xticks(range(len(ds)), [f"{d} cm" for d in ds]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tumor diameter"); ax.set_ylabel("Accuracy (LOPO, 20 phantoms)")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "acc_by_diameter.png"); plt.close(fig)

    final = fitted[best]
    if best != "physics_peak":
        imp = final.model_.feature_importances_.reshape(len(core.BANDS), core.N_ANT)
        fig, ax = plt.subplots(figsize=(7, 3.0), dpi=130)
        im = ax.imshow(imp, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(core.BANDS)), [f"{a}-{b} GHz" for a, b in core.BANDS])
        ax.set_xlabel("Antenna position (index, 5 degrees apart)")
        ax.set_title("Random Forest feature importance (final model)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

    # ---- save everything needed for reproducible predictions ----
    bundle = {
        "method": best, "method_name": METHODS[best], "estimator": final,
        "feature_names": fnames, "bands_ghz": core.BANDS, "freq_ghz": [1.0, 8.0, core.N_FREQ],
        "n_antennas": core.N_ANT, "location_map": core.LOCATION_MAP, "class_names": core.CLASS_NAMES,
        "class_short": core.CLASS_SHORT, "train_phantoms": sorted(set(groups[tr])),
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "physics_offset": fitted["physics_peak"].offset_,
    }
    joblib.dump(bundle, MODELS / "s11_tumor_locator.joblib", compress=3)
    card = {k: v for k, v in bundle.items() if k != "estimator"}
    card["location_map"] = {str(k): v for k, v in core.LOCATION_MAP.items()}
    card["preprocessing"] = ("dS11 = S11(tumor scan) - S11(paired healthy reference, matched by fib_ref_id); "
                             "feature = mean |dS11| inside each 1-GHz band [low, high) for each of 72 antennas; "
                             "504 features ordered band-major. No scaling (Random Forest does not need it).")
    card["test_metrics"] = results["test"][best]
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    pd.DataFrame(pred_rows).to_csv(REPORTS / "predictions.csv", index=False)
    print("\nSaved model ->", MODELS / "s11_tumor_locator.joblib")


if __name__ == "__main__":
    main()
