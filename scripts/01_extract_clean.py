"""
Phase 2 - Extract, validate, clean and export the dataset.

Run from the project root:
    python scripts/01_extract_clean.py --raw-dir data/raw

Inputs  (data/raw/):  fd_data_s11_emp.pickle, md_list_s11_emp.pickle
Outputs (data/processed/):
    metadata_clean.csv         all 200 scans, cleaned metadata + role + pairing
    s11_features.csv/.parquet  100 tumor samples x 504 features + labels + split
    delta_s11_long.parquet     full-resolution dS11 (100 x 1001 x 72) in long format
    delta_mag_f16.npz          |dS11| (float16) used by the website for plots
    cleaning_report.json       every check and its result
  examples/ : one held-out sample as raw scans (.npz) and as a feature row (.csv)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s11ml import core  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--skip-long", action="store_true", help="skip the large long-format parquet")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "examples").mkdir(exist_ok=True)
    report: dict = {}

    # ---------------- 1. Load (big array loaded exactly once) ----------------
    md = core.load_metadata(raw / "md_list_s11_emp.pickle")
    S = core.load_s11(raw / "fd_data_s11_emp.pickle")
    print(f"Loaded S11 {S.shape} {S.dtype} ({S.nbytes/1e6:.0f} MB), metadata {md.shape}")
    assert len(md) == S.shape[0], "metadata rows must match number of scans"
    report["raw"] = {"s11_shape": list(S.shape), "s11_dtype": str(S.dtype), "metadata_rows": len(md),
                     "metadata_columns": list(md.columns)}

    # ---------------- 2. Validate raw measurements ----------------
    mag = np.abs(S)
    flat = S.reshape(S.shape[0], -1)
    report["s11_checks"] = {
        "nan_values": int(np.isnan(S).sum()),
        "inf_values": int(np.isinf(S).sum()),
        "all_zero_scans": int((mag.reshape(S.shape[0], -1).max(1) == 0).sum()),
        "exact_duplicate_scans": int(S.shape[0] - len({r.tobytes() for r in flat})),
        "max_abs_s11": float(mag.max()),
        "passive_ok_(|S11|<=1)": bool(mag.max() <= 1.0),
    }
    del mag, flat

    # ---------------- 3. Clean metadata ----------------
    df = md.copy()
    df["tum_shape"] = df["tum_shape"].replace("", "none")
    df["has_tumor"] = df["tum_diam"].notna()
    df["role"] = np.where(df["has_tumor"], "tumor_scan", "healthy_reference")
    df["fib_ref_id"] = df["fib_ref_id"].astype("Int64")
    df["tum_in_fib"] = df["tum_in_fib"].astype("Int64")
    df["array_index"] = np.arange(len(df))           # row in the S11 array
    checks = {
        "id_unique": bool(df["id"].is_unique),
        "tumor_scans": int(df["has_tumor"].sum()),
        "reference_scans": int((~df["has_tumor"]).sum()),
        # tumor fields must be all-present for tumor scans and all-missing for references
        "tumor_fields_consistent": bool(
            df.loc[df.has_tumor, ["tum_diam", "tum_x", "tum_y", "tum_z"]].notna().all().all()
            and df.loc[~df.has_tumor, ["tum_diam", "tum_x", "tum_y", "tum_z"]].isna().all().all()),
        "tum_z_values": sorted(df["tum_z"].dropna().unique().tolist()),
    }

    # ---------------- 4. Pair tumor scans with their reference BY ID ----------------
    tum = df[df.has_tumor].copy()
    by_id = df.set_index("id")
    ref = by_id.loc[tum["fib_ref_id"].astype(int).values]
    pair_ok = pd.DataFrame({
        "ref_exists": tum["fib_ref_id"].isin(df["id"]).values,
        "ref_is_healthy": ref["has_tumor"].eq(False).values,
        "same_phantom": (tum["phant_id"].values == ref["phant_id"].values),
        "same_session": (tum["n_session"].values == ref["n_session"].values),
        "same_date": (tum["date"].values == ref["date"].values),
        "same_empty_ref": (tum["emp_ref_id"].values == ref["emp_ref_id"].values),
    })
    checks["pairing"] = {c: bool(pair_ok[c].all()) for c in pair_ok}
    checks["pairing"]["each_reference_used_once"] = bool(tum["fib_ref_id"].is_unique)
    checks["pairing"]["reference_is_next_row_(notebook_assumption)"] = bool(
        (ref["array_index"].values == tum["array_index"].values + 1).all())
    if not all(checks["pairing"][k] for k in pair_ok.columns):
        raise SystemExit("Pairing validation failed - see cleaning_report.json")

    # one location per phantom? (important for leakage)
    loc_per_phantom = tum.groupby("phant_id")[["tum_x", "tum_y"]].nunique().max().max()
    checks["locations_per_phantom_max"] = int(loc_per_phantom)
    checks["label_values_valid"] = bool(
        all((x, y) in core.LOCATION_MAP for x, y in zip(tum.tum_x, tum.tum_y)))
    report["metadata_checks"] = checks

    # Map pairing back into the full metadata table
    df["paired_reference_id"] = df["fib_ref_id"]
    ref_of = dict(zip(tum["fib_ref_id"].astype(int), tum["id"]))
    df["paired_tumor_id"] = df["id"].map(ref_of).astype("Int64")
    df.to_csv(out / "metadata_clean.csv", index=False)

    # ---------------- 5. dS11 and features ----------------
    ti = tum["array_index"].values
    ri = ref["array_index"].values
    D = (S[ti] - S[ri]).astype(np.complex64)            # (100, 1001, 72)
    X = core.band_features(D)                           # (100, 504)
    y = np.array([core.LOCATION_MAP[(a, b)] for a, b in zip(tum.tum_x, tum.tum_y)])

    # sanity: rotation of features == features of rotated dS11
    assert np.allclose(core.roll_features(X[:3], 1),
                       core.band_features(np.roll(D[:3], core.QUARTER_TURN, axis=2)), rtol=1e-5)

    dmag = np.abs(D)
    sample_energy = dmag.mean(axis=(1, 2))
    z = (sample_energy - np.median(sample_energy)) / (1.4826 * np.median(np.abs(sample_energy - np.median(sample_energy))))
    report["delta_checks"] = {
        "shape": list(D.shape),
        "nan_or_inf": int((~np.isfinite(D)).sum()),
        "duplicate_feature_rows": int(len(X) - len(np.unique(X.round(12), axis=0))),
        "robust_z_outliers_(|z|>5)": tum["id"].values[np.abs(z) > 5].tolist(),
    }

    # ---------------- 6. Phantom-grouped train/test split ----------------
    # Test = one random phantom per location class + one extra from the majority class.
    rng = np.random.default_rng(SEED)
    ph_cls = tum.assign(y=y).groupby("phant_id")["y"].first()
    test_ph = []
    for c in sorted(ph_cls.unique()):
        test_ph.append(rng.choice(sorted(ph_cls[ph_cls == c].index)))
    maj = ph_cls.value_counts().idxmax()
    test_ph.append(rng.choice(sorted(set(ph_cls[ph_cls == maj].index) - set(test_ph))))
    split = np.where(tum["phant_id"].isin(test_ph), "test", "train")
    report["split"] = {"test_phantoms": sorted(test_ph), "n_train": int((split == "train").sum()),
                       "n_test": int((split == "test").sum()),
                       "phantoms_per_class": {core.CLASS_SHORT[int(c)]: int(n) for c, n in ph_cls.value_counts().items()}}

    feats = pd.DataFrame(X, columns=core.feature_names())
    info = pd.DataFrame({
        "sample_id": tum["id"].values, "reference_id": tum["fib_ref_id"].astype(int).values,
        "phant_id": tum["phant_id"].values, "session": tum["n_session"].values, "date": tum["date"].values,
        "birads": tum["birads"].values, "fib_ang": tum["fib_ang"].values,
        "tum_diam_cm": tum["tum_diam"].values, "tum_x_cm": tum["tum_x"].values,
        "tum_y_cm": tum["tum_y"].values, "tum_z_cm": tum["tum_z"].values,
        "label": y, "label_name": [core.CLASS_NAMES[int(c)] for c in y], "split": split,
    })
    table = pd.concat([info, feats], axis=1)
    table.to_csv(out / "s11_features.csv", index=False)
    table.to_parquet(out / "s11_features.parquet", index=False)
    np.savez_compressed(out / "delta_mag_f16.npz", mag=dmag.astype(np.float16), sample_id=tum["id"].values)

    # Full-resolution dS11 in long format (one row per sample x frequency x antenna)
    if not args.skip_long:
        n, nf, na = D.shape
        long = pd.DataFrame({
            "sample_id": np.repeat(tum["id"].values, nf * na).astype(np.int16),
            "freq_ghz": np.tile(np.repeat(core.FREQ_GHZ.astype(np.float32), na), n),
            "antenna": np.tile(np.arange(na, dtype=np.int8), n * nf),
            "dS11_real": D.real.ravel(), "dS11_imag": D.imag.ravel(),
        })
        long.to_parquet(out / "delta_s11_long.parquet", index=False, compression="zstd")
        del long

    # ---------------- 7. Example inputs for the website (from a TEST phantom) ----------------
    ex_row = np.where(split == "test")[0][0]
    ex_id = int(tum["id"].values[ex_row])
    np.savez_compressed(ROOT / "examples" / f"example_scan_id{ex_id}.npz",
                        tumor=S[ti[ex_row]].astype(np.complex64), reference=S[ri[ex_row]].astype(np.complex64))
    feats.iloc[[ex_row]].to_csv(ROOT / "examples" / f"example_features_id{ex_id}.csv", index=False)
    report["example"] = {"sample_id": ex_id, "phant_id": tum["phant_id"].values[ex_row],
                         "true_label": core.CLASS_NAMES[int(y[ex_row])]}

    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
