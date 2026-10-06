# 🏺 Indus-Keeladi CNN Project — Civilization Link Analysis

A deep learning research pipeline that uses **Convolutional Neural Networks** to quantify the evolutionary link between the **Indus Valley Script** (~2600–1900 BCE) and **Keeladi Graffiti** (~6th century BCE–3rd century CE), bridging the 2,000-year gap that archaeologists hypothesize connects these two South Asian writing systems.

The project implements the research gaps identified in `conversation.txt`:

1. **Scale Gap** — Manual comparison covers only 4 matched signs. This CNN compares *all* 1,001 Keeladi sherds against the 49-sign Indus alphabet (45 primary + 4 Keeladi-matched) automatically.
2. **Subjectivity Gap** — Replaces "visual resemblance" judgments with mathematical probability distributions & feature-map evidence.
3. **Transformation Gap** — Maps the feature-level evolution of signs from the Indus corpus → Keeladi graffiti → Tamil-Brahmi inscriptions.
4. **Decomposition Gap** — Implements the 3×3 grid-decomposition technique described in the research sources to break compound ligatures into primary components.

---

## 📚 Documentation

| Document | What it covers |
|---|---|
| **[PRESENTATION.md](PRESENTATION.md)** | **8-slide talk outline** — title, problem statement, dataset, methodology, open-set gap, results, limitations, conclusions (includes speaker notes) |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | Architecture reference + all 14 UML diagrams, source and rendered |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Developer reference — purpose, tech stack, repository layout, data model, full module reference with every public method, execution flows, configuration constants, extension guide, design invariants, and known limitations |
| **[docs/UML_DIAGRAMS.md](docs/UML_DIAGRAMS.md)** | 14 PlantUML diagrams (system context, component, class, sequence, activity, state, deployment, package, use case) with render instructions |
| **[docs/uml/](docs/uml/)** | The same diagrams as standalone `.puml` files, ready to render in batch |
| **[docs/uml/images/](docs/uml/images/)** | Rendered diagram images (PNG) |
| **[docs/ANALYSIS_ACCURACY_ROADMAP.md](docs/ANALYSIS_ACCURACY_ROADMAP.md)** | Research analysis, failure taxonomy, and the accuracy roadmap |
| **[notebooks/](notebooks/)** | `dataset_analysis.ipynb`, `validation_report.ipynb` — EDA and publication plots |

Render the diagrams:

```powershell
python -m pip install plantuml
Get-ChildItem docs\uml\*.puml | ForEach-Object { plantuml -tsvg $_.FullName }
```

Or open any `.puml` in VS Code with the *PlantUML* extension and press `Alt+D`, or paste into
<https://www.plantuml.com/plantuml/umlviewer>.

---

## 📁 Project Structure (Dataset Layout)

```
CNN/
├── data/
│   ├── raw/
│   │   ├── indus_table_scans/              Put scanned figures from the PDF sources here
│   │   └── keeladi_graffiti_scans/         Put raw Keeladi sherd photos here
│   └── processed/
│       ├── train/                          Training data (folder name = class label)
│       │   ├── primary_core_signs/         45 classes based on Figure 65 (40 core + variants)
│       │   │   ├── sign_01_P13_Man/        P-2010 index labels from the paper
│       │   │   ├── sign_02_P60/
│       │   │   ├── ...
│       │   │   ├── sign_25_P225_Cross/     Cross shape — Keeladi match case
│       │   │   ├── ...
│       │   │   └── sign_40_P120_SemiSigns/
│       │   ├── indus_matched/              5 curated Indus signs with Keeladi match (merged by name)
│       │   │   ├── sign_41_P307/  sign_42_P318(b)  sign_43_P365  (+ sign_25 overlap)
│       │   ├── permanent_modifiers/        3 modifier classes (Figs 06-09) — demo, not used in single-head model
│       │   │   ├── mod_wedge_P200/
│       │   │   ├── mod_lining_horizontal_shedding/
│       │   │   ├── mod_lining_vertical_shedding/
│       │   │   └── mod_inclined_strokes_W02_W12_W14/
│       │   └── diacritical_marks/          7 vowel diacritics (Figure 15)
│       │       ├── dia_P128_short_stroke/
│       │       ├── dia_P127_double_stroke/
│       │       ├── dia_P147_vertical_line/
│       │       ├── dia_P341_oval/
│       │       ├── dia_P129_dual_vertical/
│       │       ├── dia_P175_curved/
│       │       └── dia_P173_converging/
│       └── val_keeladi/                    Validation / Research Gap testing
│           ├── match_Indus_225/            X-cross matched pair
│           ├── match_Indus_307/            D+line matched pair
│           ├── match_Indus_365/            V+stroke matched pair
│           ├── match_Indus_318/            Trident matched pair
│           ├── general_keeladi_graffiti/   Drop remaining 997 sherds HERE
│           └── keeladi_tamil_brahmi/       Evolution Stage 3
│               ├── inscriptions_atan/
│               ├── inscriptions_kuviran_atan/
│               └── general_brahmi_letters/
│
├── src/
│   ├── train.py                            Training pipeline + augmentation
│   ├── evaluate.py                         Keeladi matching + civilization-link report
│   ├── preprocessing/
│   │   ├── image_normalization.py          64×64 grayscale [0,1] pipeline
│   │   └── grid_decomposition.py           3×3 grid method (Figure 01)
│   └── models/
│       ├── indus_classifier_cnn.py         Single-head + Multi-head CNN architecture
│       └── weight_transfer.py              Indus → Keeladi domain adaptation
│
├── models/
│   ├── indus_classifier.keras              Trained weights
│   ├── indus_classifier_classes.txt        Class label list
│   └── evaluation_results/                 Report + visualizations
│       ├── keeladi_evaluation_report.txt   Detailed text report
│       ├── match_statistics.png            Bar/pie/summary charts
│       └── confidence_distribution.png     Prediction confidence histogram
│
├── notebooks/                              Jupyter notebooks (EDA, publication plots)
├── run_pipeline.py                         ⭐ One-click TRAIN + EVALUATE launcher
├── dashboard.py                            Streamlit interactive dashboard
├── dashboard.html                          (Optional) static HTML dashboard
├── generate_sample_data.py                 Demo synthetic-data generator
├── conversation.txt                        Full AI roadmap + project planning chat
└── requirements.txt
```

> **Note:** Folder names are the class labels. Any PNG/JPG/BMP/TIFF you drop inside is auto-picked up by the pipeline. Drop the 1,001 remaining Keeladi sherds into `data/processed/val_keeladi/general_keeladi_graffiti/` and the evaluate script will score them all.

---

## 🧠 Environment Setup

Your Python 3.11 virtual environment `indus_keeladi_env` is **already created** and fully populated with:

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | 2.21.0 | CNN framework (CPU mode — GPU needs WSL2) |
| Keras | 3.x | High-level model API |
| OpenCV | 5.0.0 | Image loading, processing, grid decomposition |
| NumPy | 2.4.6 | Tensor math |
| Pandas + Matplotlib + Seaborn | — | Reports + charts |
| Scikit-learn | 1.9.0 | Train/val splits, metrics |
| Streamlit | 1.61.1 | Interactive dashboard |
| Jupyter Lab + IPython | — | EDA notebooks |

---

## 🚀 How to Run

All commands use the environment's Python directly. From **PowerShell**, **cmd**, or any terminal with `CNN/` as your working directory:

### ✅ Option 1 — One-Click End-to-End Pipeline (Recommended)

```powershell
# Honest pipeline: loads the trained model, runs the validity audit and the
# real Keeladi evaluation, and writes MODEL-DERIVED reports (no simulated metrics)
C:\Users\Administrator\Desktop\CNN\indus_keeladi_env\Scripts\python.exe run_pipeline.py
```

Produces:
- `models/evaluation_results/validity_audit_report.txt` (+ `validity_audit.json`)
- `models/evaluation_results/keeladi_evaluation_report.txt`
- `models/evaluation_results/keeladi_predictions.json` (real per-sherd top-3)
- comparison galleries / plots in `models/evaluation_results/`

> ⚠️ This script no longer fabricates anything. Previously it built training
> curves with `np.random.normal`, drew confidences from `np.random.beta`, and
> printed a hard-coded "75.5% match rate". That code has been deleted; if the
> model is missing the pipeline stops instead of inventing results.


### ✅ Option 2 — Training Only (Full Quality Mode)

```powershell
# 80 epochs, best weights restored via EarlyStopping (patience=10)
C:\Users\Administrator\Desktop\CNN\indus_keeladi_env\Scripts\python.exe src\train.py
```

Inside [train.py](file:///C:/Users/Administrator/Desktop/CNN/src/train.py):
- 25× data augmentation (rotation, zoom, translation, contrast, noise) because the dataset only has 1-2 images per class
- Safe train/val split (handles single-sample classes by falling back to non-stratified splits)
- Adam optimizer with ReduceLROnPlateau + EarlyStopping

Adjust epochs / augmentation factor at the top of `train.py`.

### ✅ Option 3 — Evaluate Only (Score Keeladi Graffiti Against Saved Model)

```powershell
# Runs inference on all folders in data/processed/val_keeladi/
C:\Users\Administrator\Desktop\CNN\indus_keeladi_env\Scripts\python.exe src\evaluate.py
```

Inside [evaluate.py](file:///C:/Users/Administrator/Desktop/CNN/src/evaluate.py):
- **Threshold default: 50%** for research discovery (lower = more matches found)
- Returns Top-3 probability predictions **per image** so you can see 2nd/3rd best matches
- Produces a text report listing every sherd with its confidence bar chart
- Saves two PNG charts: match statistics 4-panel figure + confidence distribution

### ✅ Option 4 — Interactive Dashboard (Streamlit)

```powershell
C:\Users\Administrator\Desktop\CNN\indus_keeladi_env\Scripts\streamlit.exe run dashboard.py
```

Opens a browser tab with the live dashboard showing:
- Project overview metrics
- Training accuracy / progress
- Dataset inventory counts
- Evaluation report text + match statistics plot
- Full CNN architecture diagram
- All 49 Indus sign class labels

---

## 📊 Measured Results (honest, model-derived)

These are the numbers the current code actually produces (regenerate with
`python run_pipeline.py`). Earlier README versions quoted **fabricated** metrics
(fake training curves, `np.random` confidences, a hard-coded "75.5% match rate");
those have been removed. Full analysis: `docs/ANALYSIS_ACCURACY_ROADMAP.md`.

| Metric | Measured value | Where it comes from |
|---|---|---|
| Classes | **43** (40 core signs + 3 annexure-only) | Fig. 65 allograph merge + annexure remap |
| Training images | 980 | file count on disk |
| Validation accuracy (leak-controlled) | **0.7279** final / **0.7755** best | `src/train.py` |
| Leakage: val images with a ≥0.99 twin in train (random split) | 10.20% | `src/audit_validity.py` |
| Leakage: same, source-disjoint split | 5.00% | `src/audit_validity.py` |
| Open-set: blank image top-1 confidence | **0.1017** (was 0.9989) | `src/audit_validity.py` |
| Open-set: positive (real Indus) top-1 confidence | **0.7945** (was 0.5602) | `src/audit_validity.py` |
| Keeladi images analysed | 37 | `src/evaluate.py` |
| Mean prediction confidence | 0.4787 | `src/evaluate.py` |
| Match rate @ 0.5 threshold | 35.14% (13/37) | `src/evaluate.py` |
| Known-pair top-1 correctness | **0/4** | `src/evaluate.py` + audit |
| Verification permutation p-value | **0.264** (not significant) | `src/audit_validity.py` |

### Sign matching (open-set verification) — `src/sign_matcher.py`

The four hand-paired sherds are now scored as a **verification** task, not a
43-way softmax: each sherd is segmented into glyphs (pot outline excluded), then
matched by CNN embedding similarity + dilation-tolerant stroke coverage.

| sherd | old softmax "match" | **verification score** | rank /43 |
|---|---|---|---|
| match_Indus_225 | ~0.47 | **63.96%** | **4** ✅ top-5 |
| match_Indus_307 | ~0.44 | **64.73%** | 23 |
| match_Indus_318 | ~0.50 | **71.49%** | 9 |
| match_Indus_365 | ~0.46 | **63.11%** | 16 |

mean expected **65.82%** vs random-class **57.36%** · 1/4 in top-5 ·
permutation **p = 0.1025** (not significant at 0.05; n = 4).

The scores rose because the old path was classifying the *whole potsherd photo*
(auto-crop grabs the pot outline, not the glyph) and reporting a softmax as a
similarity. Caveat: a high score means the **shapes agree**, not that scripts are
related.

### 🔎 What this means (the honest finding)

The CNN now **works properly on Indus signs** (0.79 confidence on real sign
images) and **correctly rejects non-signs** (blank/noise ≈ 0.10–0.19, i.e. it
can finally say "this is not a sign" — the old model scored blanks at 0.999).
That improvement came from fixing the *labels*, not the network: Figure 65 shows
serials 18/21/28/30 carry several P-numbers joined by "or" (one sign, several
allographs), and the Keeladi annexure numbers its signs with **Mahadevan**, not
P-2010 — so two "expected matches" were pointing at entirely different glyphs.

Even so, the model does **not** reproduce the 4 hand-published Indus↔Keeladi
correspondences (0/4, permutation p = 0.264). With n = 4 and synthetic training
data that proves nothing about the archaeology either way — it says the current
data cannot test the claim.

**Next high-value step:** digitise Fig. 65 (40 signs) and the Fig. 59 allograph
plate from `docs/THE INDUS SCRIPT …pdf` (it lists NFM Unicode PUA codepoints per
sign) to replace the synthetic clones with real allographic variety.

---

## 🧠 How the Research Gap is Addressed (status-honest)

| Gap | Component | Status |
|---|---|---|
| **Scale** | `evaluate.py` scores every sherd non-interactively | implemented (37 sherds scored) |
| **Subjectivity** | `predict_keeladi_matches()` returns probability distributions | implemented |
| **Transformation** | `WeightTransfer` progressive unfreeze + domain adaptation | scaffolded; not yet run on real data |
| **Decomposition** | `GridDecomposer` 3×3 density/symmetry features | implemented but **not fed to the model** |
| **Validity (new)** | `src/audit_validity.py` leakage / open-set / verification | implemented, and it *fails* the current setup |


---

## 🔬 Research Workflow (What to Do Next)

1. **Populate training data** — digitize all 40 core signs from **Figure 65** and their variants from **Figure 59** (identical signs by engraving style) → drop 10–20 images per `sign_XX_*` folder. `generate_sample_data.py` now auto-expands 1 image → 20 realistic variants as a bootstrap (re-run it after adding new scans).
2. **Populate Keeladi** — drop the remaining 997 Keeladi graffiti sherd images into `data/processed/val_keeladi/general_keeladi_graffiti/`.
3. **Populate Tamil-Brahmi** — drop 56 inscribed sherd images into the `keeladi_tamil_brahmi/` subfolders.
4. **Run `run_pipeline.py` with 100 epochs** (edit epochs inside) — grab a coffee while it trains.
5. **Open the report** at `models/evaluation_results/keeladi_evaluation_report.txt` — you'll see **Top-3 Indus matches per sherd**.
6. **For publications**, run notebooks in `notebooks/` to generate EDA figures on allograph diversity, feature-map t-SNE, and confidence heatmaps.

---

## ⚙️ Hardware Notes

- TensorFlow 2.21 on **native Windows runs CPU only** (GPU mode disabled by Google since 2.11).
- For GPU training, move to WSL2 + CUDA or install tensorflow-directml-plugin.
- Current training on i7-class CPU: ~4 min / 20 epochs with 20× augmentation → ~16 min for 80 epochs.

---

## 🗂️ File Quick-Reference

| Want to... | Open |
|---|---|
| Change epochs / batch size / augmentation | [train.py](file:///C:/Users/Administrator/Desktop/CNN/src/train.py#L321-L373) |
| Change confidence threshold for matches | [evaluate.py](file:///C:/Users/Administrator/Desktop/CNN/src/evaluate.py#L503-L505) |
| Change CNN architecture (layers, filters) | [indus_classifier_cnn.py](file:///C:/Users/Administrator/Desktop/CNN/src/models/indus_classifier_cnn.py#L30-L164) |
| Change image size (64→128 etc.) | All modules use `ImageNormalizer(target_size=...)` |
| See the AI roadmap / original conversation | [conversation.txt](file:///C:/Users/Administrator/Desktop/CNN/conversation.txt) |

---

## 📚 References

The pipeline implements methods outlined in the three research sources documented in `conversation.txt`:
1. **The Indus Script — Recognition as an Alphabet** — core 40-sign alphabet, 3×3 grid decomposition technique, modifier + diacritic system (Figures 01–65).
2. **Keeladi Excavation Reports** — 1,001 graffiti sherds + 56 Tamil-Brahmi inscribed sherds, direct 4-sign comparison table (Page 58).
3. **Allographic Variety Table** (Figure 59) — used for style-invariant training targets.
