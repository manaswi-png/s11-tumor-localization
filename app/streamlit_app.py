"""
S11-Based Tumor Localization - research demo website.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s11ml import core  # noqa: E402
from s11ml.methods import METHODS  # noqa: E402,F401  (needed to unpickle the model)

PROC, MODELS, REPORTS, FIG, EX = (ROOT / "data" / "processed", ROOT / "models", ROOT / "reports",
                                  ROOT / "reports" / "figures", ROOT / "examples")
DISCLAIMER = ("**Research demo only.** This is an experimental model trained on 100 measurements of "
              "20 breast phantoms (lab models, not patients). It is **not** a clinically validated "
              "diagnostic tool and must not be used for medical decisions.")

st.set_page_config(page_title="S11 Tumor Localization", page_icon="📡", layout="wide")


# ----------------------------------------------------------------------------- loaders (cached)
@st.cache_resource
def load_model():
    return joblib.load(MODELS / "s11_tumor_locator.joblib")


@st.cache_data
def load_features():
    return pd.read_parquet(PROC / "s11_features.parquet")


@st.cache_data
def load_metadata():
    return pd.read_csv(PROC / "metadata_clean.csv")


@st.cache_data
def load_json(path: str):
    return json.loads(Path(path).read_text())


@st.cache_data
def load_delta_mag():
    z = np.load(PROC / "delta_mag_f16.npz")
    return z["mag"], z["sample_id"]


def fig_to_st(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ----------------------------------------------------------------------------- shared plots
def antenna_angles_deg(offset: float) -> np.ndarray:
    """Physical azimuth of each antenna, derived from the data: peak for (-,-) (225 deg) sits at
    index `offset`, and the angle decreases by 5 deg per antenna index."""
    return 225.0 - core.ANT_STEP_DEG * (np.arange(core.N_ANT) - offset)


def geometry_figure(features_row: np.ndarray, pred: int, offset: float, true: int | None = None):
    prof = core.angular_profile(features_row)[0]
    prof_n = prof / prof.max()
    ang = np.deg2rad(antenna_angles_deg(offset))
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=130)
    ax.add_patch(plt.Circle((0, 0), 7.0, fill=True, color="#f3e9e2", ec="#c9a58f", lw=1.2))
    r_ring = 8.2
    sc = ax.scatter(r_ring * np.cos(ang), r_ring * np.sin(ang), c=prof_n, cmap="inferno_r", s=28, vmin=0, vmax=1,
                    edgecolors="none", zorder=3)
    # radial response curve
    rr = r_ring + 2.2 * prof_n
    ax.plot(np.r_[rr * np.cos(ang), rr[0] * np.cos(ang[0])], np.r_[rr * np.sin(ang), rr[0] * np.sin(ang[0])],
            color="#8a3b12", lw=1)
    for c, (x, y) in core.CLASS_COORDS.items():
        is_pred = c == pred
        ax.scatter(x, y, s=260 if is_pred else 120, marker="o",
                   color="#d1495b" if is_pred else "white", edgecolors="#333", lw=1.2, zorder=4)
        if true is not None and c == true:
            ax.scatter(x, y, s=520, facecolors="none", edgecolors="#2e7d32", lw=2.2, zorder=5)
    ax.axhline(0, color="#bbb", lw=0.6, zorder=1); ax.axvline(0, color="#bbb", lw=0.6, zorder=1)
    ax.set_xlim(-11.5, 11.5); ax.set_ylim(-11.5, 11.5); ax.set_aspect("equal")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title("Top view: antenna ring coloured by |ΔS11|\nred dot = predicted"
                 + (", green ring = true" if true is not None else ""), fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="relative |ΔS11|")
    fig.tight_layout()
    return fig


def proba_chart(P: np.ndarray):
    dfp = pd.DataFrame({"location": [f"{core.CLASS_NAMES[c].split(' (')[0]} {core.CLASS_SHORT[c]}" for c in range(4)],
                        "probability": P})
    st.bar_chart(dfp, x="location", y="probability", horizontal=True, height=220)


# ----------------------------------------------------------------------------- prediction core
def predict_rows(bundle, X: np.ndarray):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    est = bundle["estimator"]
    P = est.predict_proba(X)
    pred = P.argmax(1)
    phys = core.physics_predict(X, bundle["physics_offset"])
    return pred, P, phys


def explain_result(pred, P, phys, true=None, in_training=False):
    conf = float(P.max())
    name = core.CLASS_NAMES[int(pred)]
    x, y = core.CLASS_COORDS[int(pred)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted location", core.CLASS_SHORT[int(pred)], help=name)
    c2.metric("Model vote share", f"{conf:.0%}",
              help="Fraction of the 300 trees that voted for this class. Not a calibrated probability.")
    c3.metric("Physics cross-check", "agrees" if phys == pred else "DISAGREES")
    st.markdown(f"**Result:** the tumor is most likely in the **{name}** position "
                f"(x = {x:+.2f} cm, y = {y:+.2f} cm, z = −6.5 cm).")
    notes = []
    if conf < 0.6:
        notes.append("Vote share is below 60 % — the trees are split; treat this prediction as uncertain.")
    if phys != pred:
        notes.append(f"The no-ML physics rule (strongest antenna) points to **{core.CLASS_SHORT[int(phys)]}** instead. "
                     "Disagreement is a warning sign.")
    if true is not None:
        ok = int(true) == int(pred)
        notes.append(("✅ Correct" if ok else "❌ Wrong") + f" — true location is {core.CLASS_NAMES[int(true)]}.")
        if in_training:
            notes.append("⚠️ This phantom was **in the training set**, so this is not an independent test. "
                         "Pick a sample marked *test* for a fair check.")
    for n in notes:
        st.info(n)


# ============================================================================= pages
def page_overview():
    st.title("S11-Based Tumor Localization with Machine Learning")
    st.warning(DISCLAIMER)
    m = load_json(str(REPORTS / "metrics.json"))
    sel = m["selected_method"]
    t = m["test"][sel]
    c = st.columns(4)
    c[0].metric("Test accuracy (5 unseen phantoms)", f"{t['accuracy']:.0%}")
    c[1].metric("Test macro F1", f"{t['macro_f1']:.2f}")
    c[2].metric("Mean localization error", f"{t['mean_localization_error_cm']:.2f} cm")
    c[3].metric("LOPO accuracy (all 20 phantoms)", f"{m['lopo_all_20'][sel]['accuracy']:.0%}")

    st.subheader("What the project does")
    st.markdown("""
A monostatic antenna rotates around a breast phantom and measures the **reflection coefficient S11** at
**72 positions** (every 5°) and **1001 frequencies** (1–8 GHz). A tumor has different dielectric properties
from the surrounding tissue, so it slightly changes the reflected signal — mostly at the antennas closest to it.

**Task:** given a tumor scan and the matching healthy scan of the same phantom, predict **which of 4 positions**
the tumor is in. This is **classification** (4 discrete classes), not continuous coordinate regression:
the dataset only contains 4 distinct (x, y) positions at a single height z = −6.5 cm, so a regression model
would have nothing between those points to learn from.
""")
    st.subheader("Method")
    st.markdown(r"""
1. **Pair** each tumor scan with its healthy reference using the metadata field `fib_ref_id` (verified: same phantom, session, date).
2. **Tumor signature:** $\Delta S_{11}(f, a) = S_{11}^{\text{tumor}}(f,a) - S_{11}^{\text{reference}}(f,a)$ — the background cancels, the tumor echo remains.
3. **Features (504):** mean $|\Delta S_{11}|$ in 7 one-GHz bands × 72 antennas.
4. **Model:** Random Forest (300 trees). Each tree learns yes/no rules such as *"is band 2–3 GHz at antenna 17 above a threshold?"*; the forest votes.
5. **Rotation augmentation (training only):** shifting the antenna axis by 18 positions = rotating the tumor by 90°, so each training scan also yields 3 rotated copies. This gives every class examples from every phantom.
6. **Validation without leakage:** phantoms never appear in both training and evaluation. Validation = leave-one-phantom-out (LOPO) on 15 training phantoms; final test = 5 held-out phantoms, evaluated once.
""")
    st.subheader("Key finding about leakage")
    st.markdown(f"""
Each phantom was scanned with its tumor in **only one position**, so a random train/test split puts
near-identical scans of the same phantom on both sides. The original notebook's random split reported 100 %.
Under honest phantom-grouped testing the original Random Forest reaches **{m['test']['rf_baseline']['accuracy']:.0%}**
on the test phantoms. It **cannot predict the upper-right (+,+) class at all**, because only one phantom has a tumor
there. Rotation augmentation fixes this: **{t['accuracy']:.0%}** test accuracy.
""")


def page_data():
    st.title("Dataset")
    st.warning(DISCLAIMER)
    feats, meta = load_features(), load_metadata()
    rep = load_json(str(PROC / "cleaning_report.json"))
    c = st.columns(4)
    c[0].metric("Raw scans", rep["raw"]["s11_shape"][0])
    c[1].metric("Tumor samples (paired)", len(feats))
    c[2].metric("Phantoms", feats.phant_id.nunique())
    c[3].metric("Features per sample", len(core.feature_names()))

    st.subheader("Raw data structure")
    st.markdown("""
- `fd_data_s11_emp.pickle` → complex array **(200 scans, 1001 frequencies, 72 antennas)**
- `md_list_s11_emp.pickle` → 200 metadata dictionaries, one per scan (same order)
- 100 scans contain a tumor; 100 are healthy references. Each tumor scan names its reference via `fib_ref_id`.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Samples per tumor location**")
        cnt = feats.groupby("label_name").size().rename("samples").reset_index()
        st.bar_chart(cnt, x="label_name", y="samples", height=260)
    with col2:
        st.markdown("**Phantoms per tumor location** (what limits learning)")
        pc = feats.groupby("label_name").phant_id.nunique().rename("phantoms").reset_index()
        st.bar_chart(pc, x="label_name", y="phantoms", height=260, color="#d1495b")

    st.subheader("Data-quality checks")
    st.json(rep, expanded=False)
    st.markdown("""
**Cleaning decisions**
- **No rows removed:** no NaN/Inf values, no duplicate scans, every |S11| ≤ 1 (physically valid for a passive antenna), no outlier samples (robust z-score > 5).
- **Missing tumor fields on reference scans are expected**, not errors (a healthy scan has no tumor diameter). They are not imputed.
- Empty `tum_shape` strings → `none`. `fib_ref_id` stored as an integer.
- **Pairing by ID, not row order**, then verified (same phantom/session/date/empty-chamber reference; each reference used once).
""")

    st.subheader("Explore one sample")
    mag, sid = load_delta_mag()
    opts = feats.sample_id.tolist()
    s = st.selectbox("Sample", opts, format_func=lambda i: (
        f"id {i} · {feats.set_index('sample_id').loc[i,'phant_id']} · "
        f"{feats.set_index('sample_id').loc[i,'tum_diam_cm']} cm · "
        f"{core.CLASS_SHORT[int(feats.set_index('sample_id').loc[i,'label'])]}"))
    k = int(np.where(sid == s)[0][0])
    row = feats[feats.sample_id == s].iloc[0]
    a, b = st.columns([3, 2])
    with a:
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=120)
        im = ax.imshow(mag[k].astype(float), aspect="auto", origin="lower", extent=[0, 72, 1, 8], cmap="magma")
        ax.set_xlabel("Antenna position (index)"); ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"|ΔS11| for sample {s} — tumor {row.tum_diam_cm} cm at ({row.tum_x_cm:+}, {row.tum_y_cm:+})", fontsize=9)
        fig.colorbar(im, ax=ax, label="|ΔS11|"); fig.tight_layout(); fig_to_st(fig)
    with b:
        prof = pd.DataFrame({"antenna": np.arange(72),
                             "mean |ΔS11|": core.angular_profile(row[core.feature_names()].to_numpy(float))[0]})
        st.markdown("**Angular profile** (mean over all bands) — the peak faces the tumor")
        st.line_chart(prof, x="antenna", y="mean |ΔS11|", height=300)

    st.subheader("Average angular profile by location")
    prof_all = core.angular_profile(feats[core.feature_names()].to_numpy())
    dfp = pd.DataFrame(prof_all).assign(loc=feats.label.map(core.CLASS_SHORT).values).groupby("loc").mean().T
    dfp.index.name = "antenna"
    st.line_chart(dfp, height=280)
    st.caption("Peaks are ~18 antenna positions (90°) apart: the physical basis for rotation augmentation.")

    st.subheader("Feature table (first rows)")
    st.dataframe(feats.head(20), width="stretch")


def page_eval():
    st.title("Model evaluation")
    st.warning(DISCLAIMER)
    m = load_json(str(REPORTS / "metrics.json"))
    st.markdown(f"""
**Protocol.** Train phantoms ({m['protocol']['n_train']} samples): {", ".join(m['protocol']['train_phantoms'])}
**Test phantoms** ({m['protocol']['n_test']} samples, never used for training or model choice): {", ".join(m['protocol']['test_phantoms'])}
Note: the training phantoms contain **no** (+,+) tumors — only one phantom in the whole dataset has one, and it is in the test set.
Deployed model (chosen on validation, among the ML methods): **{METHODS[m['selected_method']]}**.
""")
    rows = []
    for split, key in [("Validation: LOPO on 15 train phantoms", "validation_lopo_train"),
                       ("Test: 5 unseen phantoms", "test"), ("LOPO on all 20 phantoms", "lopo_all_20")]:
        for k, name in METHODS.items():
            r = m[key][k]
            rows.append({"evaluation": split, "method": name, "n": r["n"], "accuracy": r["accuracy"],
                         "balanced acc.": r["balanced_accuracy"], "macro precision": r["macro_precision"],
                         "macro recall": r["macro_recall"], "macro F1": r["macro_f1"],
                         "mean loc. error (cm)": r["mean_localization_error_cm"]})
    r = m["leaky_random_cv_notebook"]["rf_baseline"]
    rows.append({"evaluation": "LEAKY random 5-fold CV (notebook style)", "method": METHODS["rf_baseline"], "n": r["n"],
                 "accuracy": r["accuracy"], "balanced acc.": r["balanced_accuracy"], "macro precision": r["macro_precision"],
                 "macro recall": r["macro_recall"], "macro F1": r["macro_f1"],
                 "mean loc. error (cm)": r["mean_localization_error_cm"]})
    tab = pd.DataFrame(rows)
    st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns if tab[c].dtype == float}),
                 width="stretch", hide_index=True)

    st.subheader("Confusion matrices — held-out test set")
    cols = st.columns(3)
    for c, k in zip(cols, METHODS):
        c.image(str(FIG / f"cm_test_{k}.png"), width="stretch")
    st.subheader("Confusion matrices — leave-one-phantom-out, all 20 phantoms")
    cols = st.columns(3)
    for c, k in zip(cols, METHODS):
        c.image(str(FIG / f"cm_lopo_{k}.png"), width="stretch")

    a, b = st.columns(2)
    a.subheader("Accuracy by tumor size"); a.image(str(FIG / "acc_by_diameter.png"), width="stretch")
    if (FIG / "feature_importance.png").exists():
        b.subheader("What the forest looks at"); b.image(str(FIG / "feature_importance.png"), width="stretch")

    st.subheader("Per-class results (deployed model, test set)")
    pc = pd.DataFrame(m["test"][m["selected_method"]]["per_class"]).T
    st.dataframe(pc, width="stretch")

    with st.expander("How to read these metrics"):
        st.markdown(r"""
- **Accuracy** = correct / total. Misleading with imbalanced classes (70 % of samples are (−,−)).
- **Precision** (per class) = TP / (TP + FP): when the model says "class k", how often is it right?
- **Recall** (per class) = TP / (TP + FN): of the real class-k tumors, how many were found?
- **F1** = 2·P·R / (P + R). **Macro** = plain average over classes, so rare classes count equally.
- **Balanced accuracy** = mean recall over classes.
- **Localization error** = distance between predicted and true tumor centre: 0 cm (correct), 4.5 cm (neighbouring quadrant), 6.36 cm (opposite quadrant).
- **Confusion matrix:** row = true class, column = predicted; numbers off the diagonal are mistakes.
- **LOPO** (leave-one-phantom-out): train on 19 phantoms, test on the remaining one, repeat 20 times.
""")
    st.subheader("Limitations")
    st.markdown("""
- Only **20 phantoms**, 4 positions, one height and spherical tumors. Results say nothing about real patients, other positions or shapes.
- The tumor position is tied to the phantom (one position per phantom), so the class (+,+) is represented by a single phantom.
- Test set = 25 samples: one error moves accuracy by 4 percentage points. Treat differences of 1–2 samples as noise.
- The simple physics rule performs as well as the forest. That's a strong sanity check, and it also means this
  task is easy once a reference scan is available. The ML is useful mainly as a framework that could extend to harder settings.
- Vote share is **not** a calibrated probability.
""")


def page_predict():
    st.title("Predict tumor location")
    st.warning(DISCLAIMER)
    bundle = load_model()
    feats = load_features()
    fn = core.feature_names()
    st.caption(f"Model: {bundle['method_name']} · trained on {len(bundle['train_phantoms'])} phantoms · "
               f"scikit-learn {bundle['sklearn_version']}")
    mode = st.radio("Input", ["Pick a dataset sample", "Upload raw scans (.npz)", "Upload feature CSV"],
                    horizontal=True)

    if mode == "Pick a dataset sample":
        split_filter = st.radio("Show", ["test phantoms only (fair)", "all samples"], horizontal=True)
        sub = feats if split_filter == "all samples" else feats[feats.split == "test"]
        s = st.selectbox("Sample", sub.sample_id.tolist(), format_func=lambda i: (
            lambda r: f"id {i} · phantom {r.phant_id} · {r.tum_diam_cm} cm · split={r.split}")(
            feats.set_index("sample_id").loc[i]))
        row = feats[feats.sample_id == s].iloc[0]
        X = row[fn].to_numpy(float)[None]
        pred, P, phys = predict_rows(bundle, X)
        left, right = st.columns([3, 2])
        with left:
            explain_result(pred[0], P[0], phys[0], true=int(row.label), in_training=row.phant_id in bundle["train_phantoms"])
            st.markdown("**Vote share per location**"); proba_chart(P[0])
        with right:
            fig_to_st(geometry_figure(X[0], int(pred[0]), bundle["physics_offset"], true=int(row.label)))

    elif mode == "Upload raw scans (.npz)":
        st.markdown("""Upload a NumPy `.npz` file with **either** arrays `tumor` and `reference`
(each complex, shape **1001 × 72**: frequencies 1–8 GHz × 72 antenna positions, same system and phantom)
**or** a single array `delta` (= tumor − reference).""")
        ex = sorted(EX.glob("example_scan_*.npz"))
        if ex:
            st.download_button("Download an example .npz (held-out test phantom)", ex[0].read_bytes(), ex[0].name)
        up = st.file_uploader("Scan file", type=["npz"])
        if up is not None:
            try:
                z = np.load(io.BytesIO(up.getvalue()), allow_pickle=False)
                if "delta" in z.files:
                    D = z["delta"]
                elif {"tumor", "reference"} <= set(z.files):
                    D = core.delta_s11(z["tumor"], z["reference"])
                else:
                    raise ValueError(f"Expected arrays 'tumor' + 'reference' or 'delta'; found {z.files}")
                if D.shape != (core.N_FREQ, core.N_ANT):
                    raise ValueError(f"Expected shape (1001, 72), got {D.shape}")
                if not np.isfinite(D).all():
                    raise ValueError("Array contains NaN or Inf values")
                X = core.band_features(D)[None]
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read file: {e}")
                return
            pred, P, phys = predict_rows(bundle, X)
            left, right = st.columns([3, 2])
            with left:
                explain_result(pred[0], P[0], phys[0])
                st.markdown("**Vote share per location**"); proba_chart(P[0])
            with right:
                fig_to_st(geometry_figure(X[0], int(pred[0]), bundle["physics_offset"]))

    else:
        st.markdown(f"Upload a CSV with the **{len(fn)} feature columns** (names like `{fn[0]}` … `{fn[-1]}`). "
                    "Extra columns are ignored; one prediction per row.")
        ex = sorted(EX.glob("example_features_*.csv"))
        if ex:
            st.download_button("Download an example feature CSV", ex[0].read_bytes(), ex[0].name)
        up = st.file_uploader("Feature CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                missing = [c for c in fn if c not in df.columns]
                if missing:
                    raise ValueError(f"{len(missing)} feature columns missing, e.g. {missing[:3]}")
                X = df[fn].to_numpy(float)
                if not np.isfinite(X).all() or (X < 0).any():
                    raise ValueError("Features must be finite and non-negative (they are magnitudes)")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read file: {e}")
                return
            pred, P, phys = predict_rows(bundle, X)
            out = pd.DataFrame({"row": range(len(X)), "predicted": [core.CLASS_NAMES[int(c)] for c in pred],
                                "vote_share": P.max(1).round(3),
                                "physics_rule": [core.CLASS_SHORT[int(c)] for c in phys],
                                "agree": pred == phys})
            st.dataframe(out, width="stretch", hide_index=True)
            i = st.number_input("Show details for row", 0, len(X) - 1, 0)
            left, right = st.columns([3, 2])
            with left:
                explain_result(pred[i], P[i], phys[i])
                proba_chart(P[i])
            with right:
                fig_to_st(geometry_figure(X[i], int(pred[i]), bundle["physics_offset"]))

    with st.expander("How the website makes this prediction"):
        st.markdown("""
1. Raw scans → ΔS11 = tumor − reference (1001 × 72 complex numbers).
2. ΔS11 → 504 features: mean |ΔS11| in each 1-GHz band for each antenna (same function used in training: `s11ml/core.py`).
3. The saved Random Forest (`models/s11_tumor_locator.joblib`) votes; the location with most votes wins.
4. Independently, the physics rule finds the antenna with the strongest response and maps its angle to a quadrant.
""")


def page_downloads():
    st.title("Downloads")
    st.warning(DISCLAIMER)
    files = [
        (PROC / "s11_features.csv", "Cleaned ML dataset: 100 samples × 504 features + labels + split (CSV)"),
        (PROC / "s11_features.parquet", "Same dataset (Parquet)"),
        (PROC / "metadata_clean.csv", "Cleaned metadata for all 200 scans, with pairing"),
        (PROC / "cleaning_report.json", "All data-quality checks"),
        (REPORTS / "metrics.json", "All evaluation metrics"),
        (REPORTS / "predictions.csv", "Every evaluation prediction (test + LOPO)"),
        (MODELS / "model_card.json", "Model card: preprocessing, features, label map, versions"),
    ]
    for p, desc in files:
        if p.exists():
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{p.name}** — {desc} ({p.stat().st_size/1e3:,.0f} kB)")
            c2.download_button("Download", p.read_bytes(), p.name, key=p.name)
    big = PROC / "delta_s11_long.parquet"
    if big.exists():
        st.markdown(f"**{big.name}** — full-resolution ΔS11 (100 × 1001 × 72, long format: sample_id, freq_ghz, "
                    f"antenna, real, imag) ({big.stat().st_size/1e6:,.0f} MB)")
        if st.button("Prepare large file"):
            st.download_button("Download full ΔS11 parquet", big.read_bytes(), big.name, key="big")


PAGES = {"Overview": page_overview, "Dataset": page_data, "Model evaluation": page_eval,
         "Predict": page_predict, "Downloads": page_downloads}
with st.sidebar:
    st.markdown("### 📡 S11 Tumor Localization")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Experimental research demo · not for clinical use")
PAGES[choice]()
