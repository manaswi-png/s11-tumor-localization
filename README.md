# S11-Based Tumor Localization Using Machine Learning

Predict **which of four positions** a tumor occupies inside a breast phantom from microwave
reflection measurements (S11), using a Random Forest with physics-based rotation augmentation,
honest **phantom-grouped** evaluation, and a Streamlit website.

> **Research demo only.** Trained on 100 measurements of 20 lab phantoms. It is **not** a clinically
> validated diagnostic tool and must not be used for medical decisions.

---

## 1. Results (real numbers produced by `scripts/02_train_evaluate.py`)

| Evaluation | Method | Accuracy | Macro F1 | Mean loc. error |
|---|---|---|---|---|
| **Test – 5 unseen phantoms (25 samples)** | A. RF (original notebook) | 0.760 | 0.621 | 1.15 cm |
| | **B. RF + rotation augmentation (deployed)** | **0.960** | **0.960** | **0.18 cm** |
| | C. Physics rule, no ML | 1.000 | 1.000 | 0.00 cm |
| LOPO – all 20 phantoms (100 samples) | A. RF (original) | 0.930 | 0.670 | 0.32 cm |
| | B. RF + rotation aug. | 0.980 | 0.969 | 0.09 cm |
| | C. Physics rule | 0.990 | 0.964 | 0.04 cm |
| Random 5-fold CV (**leaky**, notebook style) | A. RF (original) | 0.960 | – | – |

* The original RF **never** predicts the upper-right (+,+) class correctly under grouped testing
  (recall 0/5), because only one phantom (A2F1) contains a tumor there. Its 93 % accuracy hides that.
* Model choice was made on validation (LOPO inside the 15 training phantoms: A 0.893, B 0.973, C 0.987)
  **before** looking at the test set. The test set was evaluated once.
* 25 test samples → one error = 4 percentage points. B and C differ by one sample: treat as equal.

## 2. Project structure

```
s11-tumor-localization/
├── README.md                     this file
├── requirements.txt
├── .streamlit/config.toml        website theme
├── s11ml/                        shared code (training AND website use it)
│   ├── core.py                   frequency axis, bands, feature extraction, labels, rotation, physics rule
│   └── methods.py                model wrapper (RF / RF+aug / physics)
├── scripts/
│   ├── 01_extract_clean.py       Phase 2: validate, pair, clean, export
│   └── 02_train_evaluate.py      Phase 3: grouped validation, test, save model
├── data/
│   ├── raw/                      PUT THE TWO .pickle FILES HERE (not included in the zip)
│   └── processed/                cleaned CSV/Parquet, metadata, cleaning report
├── models/
│   ├── s11_tumor_locator.joblib  model + feature names + label map + physics offset + versions
│   └── model_card.json           human-readable version of the above + test metrics
├── reports/                      metrics.json, predictions.csv, figures/, screenshots/
├── examples/                     sample inputs for the website's upload page
├── app/streamlit_app.py          the website
└── notebooks/                    your original notebook (unchanged)
```

## 3. Setup and run (exact commands)

```bash
# 1. unzip and enter the folder
cd s11-tumor-localization

# 2. create a virtual environment (recommended)
python -m venv .venv
# Windows:    .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. install libraries
pip install -r requirements.txt

# 4. (only needed to regenerate data/model) copy the raw files into data/raw/
#    data/raw/fd_data_s11_emp.pickle   and   data/raw/md_list_s11_emp.pickle

# 5. regenerate the cleaned dataset (~10 s)
python scripts/01_extract_clean.py

# 6. retrain and evaluate (~1 min)
python scripts/02_train_evaluate.py

# 7. start the website  ->  opens http://localhost:8501
streamlit run app/streamlit_app.py
```

The zip already contains the processed data and the trained model, so **step 7 works without steps 4–6**.
Steps 5–6 recreate everything bit-for-bit (fixed random seed 42).
`01_extract_clean.py --skip-long` skips the 62 MB full-resolution ΔS11 Parquet file.

## 4. The data

| File | Content |
|---|---|
| `fd_data_s11_emp.pickle` | complex128 array **(200, 1001, 72)** = 200 scans × 1001 frequencies (1–8 GHz) × 72 antenna positions (5° apart) |
| `md_list_s11_emp.pickle` | list of 200 dicts (one per scan, same order): `id`, `phant_id`, `tum_diam`, `tum_x/y/z`, `fib_ref_id`, `n_session`, `date`, … |

* 100 scans contain a tumor, 100 are healthy references. Each tumor scan names its reference in `fib_ref_id`.
* Pairing was done **by ID** and verified: same phantom, session, date and empty-chamber reference;
  each reference used exactly once. (The notebook's "next row" assumption happens to hold too.)
* 20 phantoms × 5 tumor diameters (1.0–3.0 cm). **Each phantom has a single tumor position.**
  Phantoms per position: (−,−) 14, (−,+) 3, (+,−) 2, (+,+) 1.

**Cleaning decisions** (all checks in `data/processed/cleaning_report.json`)
1. Nothing was removed: no NaN/Inf, no duplicate scans, |S11| ≤ 1 everywhere (physically valid), no outliers.
2. Empty tumor fields on reference scans are *meaningful* (no tumor), so they were not imputed.
3. Empty `tum_shape` → `none`; `fib_ref_id` → integer.

**Exports**
* `s11_features.csv` / `.parquet`: 100 rows: IDs, phantom, diameter, x/y/z, `label`, `split`, + 504 features
* `metadata_clean.csv`: all 200 scans with role and pairing columns
* `delta_s11_long.parquet`: full-resolution ΔS11 (7.2 M rows: sample_id, freq_ghz, antenna, real, imag)

## 5. How it works (beginner-friendly)

**Step 1, background removal.** $\Delta S_{11}(f,a) = S_{11}^{tumor}(f,a) - S_{11}^{reference}(f,a)$.
Antenna and phantom effects are the same in both scans and cancel, leaving the tumor's echo.

**Step 2, features.** For each of 7 bands (1–2, …, 7–8 GHz) and each of the 72 antennas:
$x_{b,a} = \text{mean}_{f \in b} |\Delta S_{11}(f,a)|$ → 7 × 72 = **504 numbers** per sample.

**Step 3, classification vs. regression.** The dataset only has 4 distinct (x, y) positions and one z.
Predicting one of 4 classes is **classification**. A regression model would output continuous
coordinates, but with only 4 training positions it could not learn anything in between. We still report
a distance metric: **localization error** = distance between predicted and true centre
(0 cm correct, 4.5 cm neighbouring quadrant, 6.36 cm opposite quadrant).

**Step 4, Random Forest.** A *decision tree* asks yes/no questions (e.g. "is feature `band2-3GHz_ant17` > 0.003?")
and splits the data so each branch gets purer (measured by *Gini impurity* $G = 1-\sum_k p_k^2$).
A *forest* trains 300 trees on random resamples of the data and random subsets of features, then they vote.
**Vote share** = fraction of trees voting for the winner. It is a rough confidence, not a calibrated probability.

**Step 5, rotation augmentation (the key improvement).** The strongest |ΔS11| appears at the antenna
facing the tumor: peaks at antenna ≈0 (−,−), ≈18 (−,+), ≈36 (+,+), ≈54 (+,−). They are 18 positions = 90° apart.
So circularly shifting the 72-antenna axis by 18 is the same as rotating the tumor by 90°:
(x, y) → (y, −x). Each training sample gives 3 extra rotated copies with correct labels. Now every
class has examples from every phantom. This is applied to **training data only**, never to test data.

**Step 6, physics rule (sanity check).** Find the antenna with the largest average |ΔS11| and map its
angle to the nearest quadrant (one offset learned from training data). It uses no ML, and it agrees with
the forest on almost everything, which gives confidence that the forest learned real physics rather than artefacts.

**Step 7, why grouped splitting.** Scans from the same phantom are near-identical apart from tumor size.
With a random split, the model is tested on phantoms it has already seen, so it can recognise the
phantom instead of locating the tumor. That's *data leakage*. We keep whole phantoms together:
* **Test:** 5 phantoms (one per position + one extra), never touched during development.
* **Validation:** leave-one-phantom-out (LOPO) on the 15 training phantoms, used to choose the model.
* **Extra:** LOPO on all 20 phantoms gives a more stable estimate (every phantom tested once).

**Metrics.** Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R);
*macro* = average over classes (rare classes count equally); balanced accuracy = mean recall.
Confusion matrix: rows = true class, columns = predicted class.

## 6. The website

`streamlit run app/streamlit_app.py` → five pages:
**Overview** (goal, method, headline metrics), **Dataset** (structure, class balance, quality checks,
per-sample ΔS11 heatmap and angular profile), **Model evaluation** (all metrics, confusion matrices,
accuracy by tumor size, feature importance, limitations), **Predict** (pick a sample, upload raw
`tumor`/`reference` arrays as `.npz`, or upload a feature CSV → predicted location, vote share,
physics cross-check, top-view diagram), **Downloads**.

The Predict page loads `models/s11_tumor_locator.joblib` and calls the **same** `s11ml/core.py`
functions used in training, so preprocessing is identical. Example inputs are in `examples/`.

## 7. Limitations

* 20 phantoms, 4 positions, 1 height, spherical tumors: says nothing about patients or other positions.
* Tumor position is tied to the phantom, and (+,+) is represented by a single phantom.
* Small test set (25): differences of 1–2 samples are noise.
* Requires a matched healthy reference scan of the same phantom, which isn't available clinically.
* The frequency axis (1–8 GHz, 1001 points) is taken from the original notebook / dataset description.
