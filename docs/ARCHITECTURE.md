# Indus–Keeladi CNN — Technical Documentation

> Deep learning research pipeline that quantifies the **visual** relationship between the
> Indus Valley Script (~2600–1900 BCE) and the Keeladi graffiti corpus (6th c. BCE – 3rd c. CE),
> with an extension stage for Tamil-Brahmi inscriptions.
>
> This is the developer reference for the code in this repository.
> Research analysis, failure taxonomy and accuracy roadmap:
> [`ANALYSIS_ACCURACY_ROADMAP.md`](ANALYSIS_ACCURACY_ROADMAP.md).
> UML diagram sources: [`UML_DIAGRAMS.md`](UML_DIAGRAMS.md).

---

## 1. Purpose and Scope

The project answers one question mechanically:

> For each Keeladi potsherd glyph, which Indus sign class does the trained CNN rank highest,
> and how confident is that ranking?

Everything else (segmentation, Brahmi template matching, lexicon, NLP, plots, dashboard)
exists to make that one answer **inspectable**.

### 1.1 Research gaps addressed

| Gap | Traditional approach | This project |
|---|---|---|
| **Scale** | 4 hand-matched sign pairs | Every image in `data/processed/val/keeladi/` is scored |
| **Subjectivity** | "looks similar" | Softmax distributions + embedding retrieval + stroke coverage |
| **Transformation** | Qualitative evolution claims | Indus → Keeladi → Tamil-Brahmi staging + `WeightTransfer` scaffold |
| **Decomposition** | Manual 3×3 grid reading | `GridDecomposer` density/symmetry features (implemented, not yet fed to the model) |
| **Validity** | Unstated assumptions | `audit_validity.py` measures leakage, open-set false acceptance, permutation significance |

### 1.2 Non-goals

- The system **does not** decipher the Indus script.
- Visual / activation similarity **is not** evidence of linguistic descent.
- Tamil-Brahmi is deciphered, so its readings are real. The Indus sign meanings in
  `data/lexicon.json` are the project's *adopted working readings*, not scholarly consensus.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep learning | TensorFlow 2.21 / Keras 3 (CPU on native Windows) |
| Image processing | OpenCV (`cv2`), NumPy |
| Metrics | scikit-learn (splits, classification report, confusion matrix) |
| Plots | Matplotlib (`Agg` backend), Seaborn |
| PDF digitisation | PyMuPDF vector drawings + pixmaps |
| UI | Streamlit; Tkinter (`fullscreen_resizer.py`) |
| Annotation | Pillow |
| Analysis | Jupyter Lab |

Environment: in-repo virtual environment `indus_keeladi_env/`.

---

## 3. Repository Layout

```
CNN/
├── run_pipeline.py                  ⭐ Orchestrator (6 steps, honesty-gated)
├── process_model.py                 Results reader (prints artefacts, generates nothing)
├── dashboard.py                     Streamlit UI (5 sections)
│
├── src/
│   ├── train.py                     IndusKeeladiTrainer — data loading + training
│   ├── evaluate.py                  KeeladiEvaluator — closed-set scoring + reports
│   ├── audit_validity.py            run_audit — leakage / open-set / verification
│   ├── sign_matcher.py              Open-set retrieval (embedding + stroke coverage)
│   ├── siamese_embed.py             Metric learning (triplet loss) on Fig.65 glyphs
│   ├── digitise_fig65.py            Extract real glyphs from the source PDF
│   ├── decoding.py                  InscriptionDecoder — segment → dual-read → NLP
│   ├── preprocessing/
│   │   ├── image_normalization.py   ImageNormalizer (autocrop → deskew → 64×64 → [0,1])
│   │   └── grid_decomposition.py    GridDecomposer (3×3, Fig. 01 logic)
│   └── models/
│       ├── indus_classifier_cnn.py  IndusClassifierCNN (single-head + 3-head)
│       └── weight_transfer.py       WeightTransfer (progressive unfreeze, domain gap)
│
├── cnn_potsherd_annotator.py        Renders annotated potsherd images
├── cnn_annotation_generator.py      Synthetic annotated potsherds
├── fullscreen_resizer.py            Tkinter GUI for tuning annotation boxes
├── secret_resizer.py                Persists box coordinates (.secret_resizer.json)
├── reorganize_brahmi.py             One-off Brahmi reference folder reshuffle
│
├── data/
│   ├── raw/                         Source scans
│   ├── processed/
│   │   ├── train/                   Folder name == class label
│   │   │   ├── primary_core_signs/  45 folders, 880 images
│   │   │   ├── indus_matched/       5 folders, 100 images (merged by class name)
│   │   │   ├── diacritical_marks/   8 folders, 5 images
│   │   │   ├── permanent_modifiers/ 3 folders, 1 image
│   │   │   └── fig65_real/          39 real glyphs + manifest.json
│   │   ├── val/
│   │   │   ├── keeladi/             5 folders, 16 images
│   │   │   └── tamil_brahmi/        3 folders, 58 images (30 letter sub-folders)
│   │   ├── test/
│   │   └── lexicon.json             Indus + Brahmi readings cache
│   └── results/                     logs/ matches/ metrics/ visualizations/
│
├── models/
│   ├── indus_classifier.keras       Trained CNN (43 classes, 980 images)
│   ├── indus_classifier_classes.txt Class label list, one per line
│   ├── fig65_siamese.keras          Metric-learning model
│   └── evaluation_results/          All generated reports, JSON, PNGs
│
├── docs/
│   ├── THE INDUS SCRIPT Recognition as an Alphabet.pdf   Fig.65 source
│   ├── keeladi_indus.pdf                                 4-pair annexure
│   ├── ANALYSIS_ACCURACY_ROADMAP.md     Research analysis & roadmap
│   ├── ARCHITECTURE.md                 ← this file
│   ├── UML_DIAGRAMS.md                 ← PlantUML sources
│   └── uml/                            Standalone .puml files per diagram
│
├── notebooks/                       dataset_analysis.ipynb, validation_report.ipynb
└── requirements.txt
```
---

## 4. Data Model

### 4.1 Class-label convention

**Folder name = class label.** Any `.png/.jpg/.jpeg/.bmp/.tif/.tiff` dropped into a folder is
auto-discovered. There is no manifest for the training tree; only `fig65_real/` has a
`manifest.json`, produced by the digitiser.

### 4.2 Label numbering systems

Three numberings coexist and must not be confused:

| System | Example | Used by |
|---|---|---|
| **Fig.65 serial** | 1 … 40 | Row identity in the source table — the only 1:1 stable key |
| **M-1977 (Mahadevan)** | 225, 307, 318, 365 | The Keeladi annexure numbers its signs |
| **P-2010** | P219, P181/P187, P318 | Class folder names (`sign_24_P219`) |

Corrections applied at load time in `src/train.py`:

- `ALLOGRAPH_GROUPS` merges Fig.65 "or" rows that are **one sign** drawn several ways
  (45 folders → 40 core signs).
- `ANNEXURE_REMAP` moves the two annexure images (M-1977 numbered) into the core class
  they actually depict.
- Serials that merely *share* a P-number (`sign_12_P130` / `sign_13_P130_variant`,
  `sign_35_P341_Oval` / `sign_36_P341_Leaf`) are deliberately **not** merged.

`canonical_class(folder_name)` is the single resolver used by both training and auditing.

### 4.3 Rejection classes

Classes prefixed `zz_` are "not an Indus sign" negatives. `src/train.py`
(`generate_rejection_images`) synthesises them — 60 per family: `blank`, `noise`,
`strokes`, `blob`. `src/evaluate.py` never counts a `zz_` prediction as a match.

### 4.4 Expected-match map

`EXPECTED_MATCH_MAP` (duplicated in `src/evaluate.py` and `src/audit_validity.py`) maps a
Keeladi match folder to the **list** of acceptable Indus classes. Values are lists because
one sign can carry several P-numbers (allographs); matching any member counts as correct.

```
match_Indus_225 -> ["sign_24_P219"]
match_Indus_307 -> ["sign_18_P181_P187"]
match_Indus_318 -> ["sign_42_P318", "sign_42_P318b"]
match_Indus_365 -> ["sign_43_P365"]
```

### 4.5 Current on-disk state

| Tree | Folders | Images |
|---|---|---|
| `train/primary_core_signs` | 45 | 880 |
| `train/indus_matched` | 5 | 100 |
| `train/diacritical_marks` | 8 | 5 |
| `train/permanent_modifiers` | 3 | 1 |
| `train/fig65_real` | — | 39 |
| `val/keeladi` | 5 | 16 |
| `val/tamil_brahmi` | 3 (30 letter sub-dirs) | 58 |

The trained model has **43 output classes** over **980 training images** — well below the
1,001-sherd scale the top-level README describes. See §10 Known Limitations.
---

## 5. Module Reference

### 5.1 `run_pipeline.py` — Orchestrator

The single entry point. Six steps, each delegating to a real module. **It stops rather
than fabricate if the model is missing.**

| Step | Function | Delegates to |
|---|---|---|
| 1 | `scan_data()` | real file counts on disk |
| 2 | `step_audit()` | `src.audit_validity.run_audit` |
| 3 | `step_evaluate()` | `src.evaluate.KeeladiEvaluator` (+ `run_decoding`) |
| 4 | `step_sign_match()` | `src.sign_matcher.match_all` |
| 5 | `write_predictions_json()` | serialises step-3 output for the dashboard |
| 6 | `step_annotate()` | `cnn_potsherd_annotator.PotsherdAnnotator` (optional) |

Key paths: `PROJECT_ROOT`, `MODEL_PATH = models/indus_classifier.keras`,
`EVAL_DIR = models/evaluation_results/`.

### 5.2 `src/train.py` — `IndusKeeladiTrainer`

Module-level constants: `ALLOGRAPH_GROUPS`, `ANNEXURE_REMAP`, `REJECTION_PREFIX = "zz_"`,
`IMG_EXTS`.

| Method | Purpose |
|---|---|
| `_build_augmentation_pipeline()` | Keras geometric + photometric augmentation |
| `_numpy_stochastic_degrade()` | Per-image probabilistic noise/blur/threshold degradation |
| `_augment_dataset()` | Original + N augmented copies per image |
| `_safe_train_val_split()` | Stratified split with non-stratified fallback for 1-sample classes |
| `_safe_index_split()` | Split **before** augmenting (leakage control) |
| `load_training_data(augment, split_protocol)` | Main data loader — see split protocols below |
| `build_model()` | Instantiates the CNN for the discovered class count |
| `train_model(..., epochs, batch_size)` | Adam + `EarlyStopping` + `ReduceLROnPlateau` |
| `save_trained_model()` / `load_trained_model()` | `.keras` + `<stem>_classes.txt` pair |

Module functions: `load_fig65_real_glyphs()`, `generate_rejection_images()`,
`canonical_class()`.

**Split protocols** — the single most important design decision in this module:

| `split_protocol` | Behaviour | Consequence |
|---|---|---|
| `"leakage_controlled"` *(default)* | Split originals **first**, augment **train only** | Validation images are never augmented clones — honest accuracy |
| `"stratified"` | Augment first, split second | Legacy, leaky; kept only to reproduce old numbers. Emits a warning |

**Configurable schedule** (environment variables, so no code edits needed):

| Variable | Default | Meaning |
|---|---|---|
| `INDUS_AUG` | 25 | Augmented copies per source image |
| `INDUS_EPOCHS` | 80 | Training epochs |
| `INDUS_BATCH` | 16 | Batch size |

### 5.3 `src/preprocessing/image_normalization.py` — `ImageNormalizer`

Single preprocessing pipeline, applied **identically** to train and val.

Order (matters for accuracy):

```
load → grayscale → legacy polarity guess → denoise
     → Otsu polarity normalise (glyph BRIGHT on dark)
     → AUTOCROP to glyph contour (bbox + square pad)
     → DESKEW by image moments
     → resize to target_size → [0,1]
```

Public: `process_image(path)`, `process_array(arr)`, `process_directory(in, out)`.
Private helpers: `_normalize_polarity`, `_denoise`, `_autocrop_to_glyph`, `_deskew`.
`invert_if_dark_on_bright()` is retained for API compatibility only — the centre-vs-corner
heuristic is unreliable for sparse stroke glyphs; `_normalize_polarity()` is authoritative.

### 5.4 `src/models/indus_classifier_cnn.py` — `IndusClassifierCNN`

VGG-style CNN, input `(64, 64, 1)`.

| Block | Layers | Output |
|---|---|---|
| Input | `Input` | 64×64×1 |
| Block 1 | `Conv2D(32)×2` + `BatchNorm` + `MaxPool(2)` + `Dropout(0.25)` | 32×32×32 |
| Block 2 | `Conv2D(64)×2` + `BatchNorm` + `MaxPool(2)` + `Dropout(0.25)` | 16×16×64 |
| Block 3 | `Conv2D(128)×2` + `BatchNorm` + `MaxPool(2)` + `Dropout(0.25)` | 8×8×128 |
| Head | `Flatten` → `Dense(256)` → `Dense(128)` + `BatchNorm` + `Dropout(0.5)` | 128 |
| Output | `Dense(num_classes, softmax)` | num_classes |

Compile: `Adam(1e-3)`, `sparse_categorical_crossentropy`, metric `accuracy`.

`build_multi_head_model()` — functional variant with three softmax heads:
`sign_output` (loss weight 1.0), `modifier_output` (0.5), `diacritic_output` (0.3).

Other methods: `train()`, `predict()`, `save_model()`, `load_model()`, `get_model_summary()`.

### 5.5 `src/models/weight_transfer.py` — `WeightTransfer`

Indus → Keeladi domain adaptation. **Scaffolded, not yet run on real data.**

`transfer_convolutional_layers()`, `freeze_early_layers(n)`, `unfreeze_all_layers()`,
`progressive_unfreezing(epoch, total_epochs, unfreeze_schedule)`,
`adapt_learning_rate()`, `domain_adaptation_training()` (LR 1e-4 + augmentation),
`evaluate_domain_gap()` (feature-space distance + std ratio between domains).

### 5.6 `src/preprocessing/grid_decomposition.py` — `GridDecomposer`

3×3 decomposition from Figure 01 of the source paper.

`decompose_image()` → 9 cells · `analyze_grid_presence(threshold)` → binary matrix ·
`extract_features_from_grid()` → per-cell `mean_intensity`, `std_intensity`, `pixel_density`,
`horizontal_symmetry`, `vertical_symmetry` · `visualize_grid()` → overlay PNG.

> **Status:** implemented and tested, but the features are **not** fed to the CNN.
---

### 5.7 `src/audit_validity.py` — Validity Audit

Replaces the fabricated metrics that used to live in `run_pipeline.py` and
`process_model.py`. Three real, reproducible experiments.

| Audit | Function | Question answered |
|---|---|---|
| **1. Leakage** | `leakage_audit(X, y, groups, test_size, thresh=0.99, seed)` | How much reported "validation accuracy" is memorisation? Nearest-neighbour duplication under (a) random split and (b) source-disjoint split |
| **2. Open-set** | `openset_audit(model, class_names, normalizer, data_dir, n_pos=100)` | Can a closed-set softmax ever say "not an Indus sign"? False-acceptance rate on known negatives vs a positive control |
| **3. Verification** | `verification_audit(model, class_names, normalizer, data_dir, n_perm=2000)` | For the 4 curated pairs, does the model favour the expected sign? Top-1/top-3 plus a permutation-test p-value |

Support: `load_train_corpus()` (mirrors `src.train` exactly, including the allograph merge),
`load_class_names()`, `load_classifier()`, `predict_probs()`, `_contamination()`,
`_negative_groups()`, `write_report()`, `run_audit()`.

**Current result** (`validity_audit.json`): 43 classes, 980 images, leakage verdict `clean`
(3.40% of val images have a ≥0.99 twin under the random split), open-set mean negative
confidence 0.273 vs 0.826 positive, verification 2/4 top-1 and 3/4 top-3 with `p = 0.0`.

### 5.8 `src/evaluate.py` — `KeeladiEvaluator`

Closed-set classification path. **Its softmax output is a class posterior, not a similarity.**

| Method | Purpose |
|---|---|
| `load_keeladi_validation_set()` | Walks `data/processed/val/{keeladi,tamil_brahmi}`, preprocesses every image |
| `predict_keeladi_matches(validation_data, threshold=0.5)` | Top-3 classes + probabilities per image; honours the `zz_` rejection index |
| `analyze_civilization_link(predictions)` | Aggregate match rate, confidence stats, most-common signs, known-pair tally |
| `generate_report(...)` | `keeladi_evaluation_report.txt` |
| `run_decoding(output_dir)` | Delegates to `InscriptionDecoder`, writes `decoded/decoded_readings.txt` |
| `generate_known_pair_comparison(output_dir)` | Side-by-side Keeladi vs Indus figure with per-pair shape match % |
| `_plot_match_statistics()` / `_plot_confidence_distribution()` / `_plot_graffiti_gallery()` | 4-panel stats, confidence histogram, per-sherd galleries |
| `_save_single_sherd_comparisons()` | Per-image side-by-side comparisons |

Control folders excluded from match claims: `general_keeladi_graffiti`, `tamil_brahmi_*`.

**Known-pair shape comparison** uses dilation-tolerant bidirectional stroke coverage
(2 dilations, 3×3 kernel) between the Indus reference glyph and each sherd letter glyph,
keeping the best local match — so extra strokes elsewhere on the sherd cannot dilute the score.

### 5.9 `src/sign_matcher.py` — Open-Set Matcher

Answers the *verification/retrieval* question rather than classification, because a 43-way
softmax lands around 0.40–0.50 even for genuinely paired sherds.

Scoring pipeline:

1. **Segment** the sherd into individual glyphs (reuses `InscriptionDecoder`, which already
   excludes the pot outline).
2. **Embed** — CNN penultimate-layer vectors, cosine similarity to a per-class prototype.
3. **Stroke coverage** — symmetric, dilation-tolerant overlap against the class glyphs.
4. **Rank + calibrate** — report the expected sign's rank and a percentage.
5. **Permutation test** — is the result better than chance?

Functions: `binarize()`, `largest_component_mask()`, `stroke_coverage()`, `embed_model()`,
`embed_images()`, `build_class_references()`, `prototype_embeddings()`, `extract_glyphs()`,
`score_sherd()`, `match_all()`, `write_report()`, `main()`.
---

### 5.10 `src/decoding.py` — `InscriptionDecoder`

Four-stage dual-script decoding.

| Stage | Method | What happens |
|---|---|---|
| 1. Segment | `segment_inscription(path)` | Splits a multi-character potsherd into letter crops. The pot outline / sherd ends are detected and **reported as context** but never classified, so they can never cause a false comparison. Letters fused to the outline are recovered by subtracting the outline band (`_recover_letters`, `_projection_split`) |
| 2. Identify | `match_brahmi(glyph)` | Dilation-tolerant template match against the Tamil-Brahmi reference alphabet |
| | `indus_top3(glyph)` | Top-3 Indus CNN classes |
| 3. Decode | `decode_inscription(path)` / `decode_graffiti(img)` | Lookup in `data/lexicon.json`, compose readings with annotations |
| 4. NLP | `nlp_decode(result, corpus_key)` | Build syllables, fuzzy-match against attested Keeladi names, align with the corpus reading, compose an Indus gloss |

Module constants: `KEELADI_NAMES` (15 attested personal names), `CORPUS_READINGS`
(e.g. `inscriptions_kuviran_atan` → "kuviran atan"), `VOWEL_TOKENS`.

`visualize_decoding()` writes the annotated image.

### 5.11 `src/digitise_fig65.py` — Source Glyph Extraction

Extracts **real** sign glyphs from Figure 65 (pp. 35–36) of the source PDF.

| Function | Purpose |
|---|---|
| `table_rules(page)` | Vertical/horizontal rule positions read from PDF vector drawings, so crops sit strictly inside the cell |
| `_serial_anchors(page, serials)` | Locates the serial-number cells to anchor each row |
| `clean_crop(img)` | Erases ink runs spanning ≥85% of the cell (the table rules) then keeps the largest remaining component |
| `digitise()` | Renders at `ZOOM = 6.0`, crops each sign cell, writes PNG + `manifest.json` |
| `contact_sheet()` | Labelled grid QA artifact → `docs/figures/fig65_digitised_contact_sheet.png` |
| `debug_page()` / `debug_rects()` | Diagnostics for anchors and crop rectangles |

`FIG65_ROWS` maps serial → `(NFM Unicode PUA, M-1977, W-2015, P-2010)` for serials 1–39.
Serial 40 is a seal photograph, not a drawing, so it is correctly skipped.

`STROKE_SERIALS = {9,10,11,12,13,14,15,16,30}` — plain stroke glyphs that differ only in
stroke count/position. Shape matching against them is degenerate, so they are flagged
`is_stroke_sign` in the manifest and excluded from match claims.

### 5.12 `src/siamese_embed.py` — Metric Learning

Learns a **64-D L2-normalised embedding** with a batch-hard triplet loss plus a
classification head, on the 39 real Fig.65 glyphs with affine/noise perturbations.
Class-balanced batches (6 instances × 12 classes) guarantee every anchor has a positive and
hard negatives.

Architecture: `Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(128) → Pool → Conv2D(128) →
GlobalAveragePooling → Dense(64, "embedding") → L2 norm ("l2norm") → Dense(39, "logits")`.

Functions: `load_glyphs()`, `fit_canvas()`, `augment()`, `build_dataset()`, `build_test()`,
`build_reference()`, `make_model()`, `margin_triplet()`, `encode()`, `train()`,
`retrieval_report()`, `per_class_report()`, `main()`.

**Retrieval result** (`siamese_report.json`): top-1 0.859, top-5 0.974, mean rank 1.41
against chance 0.026 / 20.0.

**Honest limit:** held-out queries are perturbations of the *same* source glyphs used for
training. This measures robustness to geometric perturbation only — an **upper bound**, not
evidence of performance on independently engraved signs.

### 5.13 Presentation & Utility Layer

| File | Entry point | Purpose |
|---|---|---|
| `dashboard.py` | `streamlit run dashboard.py` | Streamlit UI. Sections: `render_statistics_section()`, `render_sign_matching_section()`, `render_tamil_brahmi_section()`, `render_resizer_section()`. `load_model_metrics()` reads `validity_audit.json` and returns `None` when no real result exists, so the UI says "run the pipeline" instead of showing invented values |
| `process_model.py` | `python process_model.py` | Pure results reader. Prints the audit, evaluation and prediction artefacts; prints `[missing] → run: python run_pipeline.py` otherwise |
| `cnn_potsherd_annotator.py` | `PotsherdAnnotator(data_dir, output_dir).annotate_all()` | Renders annotated potsherds with per-character red boxes. Its character-mapping table contains **authored reference readings**, not CNN inference — stated in the module docstring |
| `cnn_annotation_generator.py` | `CNNAnnotationGenerator(output_dir)` | Synthetic annotated potsherds for demo/figure generation |
| `fullscreen_resizer.py` | `FullscreenResizer()` | Tkinter GUI: click/drag/scroll to tune annotation boxes |
| `secret_resizer.py` | `python secret_resizer.py` | Persists tuned box coordinates to `.secret_resizer.json` |
| `reorganize_brahmi.py` | script | One-off reshuffle of Brahmi reference folders |
---

## 6. Execution Flows

### 6.1 Full pipeline (`run_pipeline.py`)

```
START
  │
  ├─ models/indus_classifier.keras exists? ──no──▶ log.error + sys.exit(1)  [never fabricate]
  │ yes
  ▼
[1/6] scan_data()               real file counts
  ▼
[2/6] step_audit()              ──▶ src.audit_validity.run_audit()
  │        leakage_audit / openset_audit / verification_audit
  │        └──▶ validity_audit.json + validity_audit_report.txt
  ▼
[3/6] step_evaluate()           ──▶ src.evaluate.KeeladiEvaluator
  │        load_keeladi_validation_set()
  │        predict_keeladi_matches(threshold=0.5)
  │        analyze_civilization_link() + generate_report()
  │        run_decoding()  ──▶ decoded/decoded_readings.txt   (optional, never faked)
  ▼
[4/6] step_sign_match()         ──▶ src.sign_matcher.match_all(n_perm=2000)
  │        segment → embed → stroke coverage → rank → permutation test
  │        └──▶ sign_match_report.txt / .json
  ▼
[5/6] write_predictions_json()  ──▶ keeladi_predictions.json  (dashboard input)
  ▼
[6/6] step_annotate()           ──▶ cnn_potsherd_annotator  (optional)
  ▼
"PIPELINE COMPLETE" + artefact paths + visual-only caveat
```

### 6.2 Training (`src/train.py`)

```
main()
  ▼ read env: INDUS_AUG / INDUS_EPOCHS / INDUS_BATCH
  ▼ IndusKeeladiTrainer(data_dir, model_dir, augment_factor)
  ▼ load_training_data(augment=True, split_protocol="leakage_controlled")
  │    ├─ discover class folders (primary_core_signs + indus_matched)
  │    ├─ resolve → canonical_class()  [ALLOGRAPH_GROUPS + ANNEXURE_REMAP]
  │    ├─ add fig65_real glyphs via load_fig65_real_glyphs()   [mapped by SERIAL]
  │    ├─ add zz_ rejection images via generate_rejection_images()
  │    ├─ preprocess each image with ImageNormalizer.process_image()
  │    ├─ SPLIT originals into train/val FIRST        ◄── leakage control
  │    └─ augment ONLY the train split
  ▼ build_model()          IndusClassifierCNN, num_classes = discovered
  ▼ train_model()          Adam 1e-3 + EarlyStopping(10) + ReduceLROnPlateau(5, ×0.5)
  ▼ save_trained_model("indus_classifier")
        ├──> models/indus_classifier.keras
        └──> models/indus_classifier_classes.txt
```

### 6.3 Preprocessing pipeline (per image)

```
cv2.imread
   ▼ grayscale (BGR2GRAY)
   ▼ invert_if_dark_on_bright()   legacy heuristic, API-compat only
   ▼ _denoise()                   morphological open/close, speckle removal
   ▼ _normalize_polarity()        Otsu split → glyph BRIGHT on dark bg
   ▼ _autocrop_to_glyph()         contour bbox + square pad  ◄── biggest accuracy gain
   ▼ _deskew()                    image moments, correct shear
   ▼ resize to (64, 64)           INTER_AREA
   ▼ astype(float32) / 255.0      → [0, 1]
```

---
## 7. Configuration Reference

### 7.1 Environment variables

| Variable | Default | Read by | Meaning |
|---|---|---|---|
| `INDUS_AUG` | 25 | `src/train.py:734` | Augmented copies per source image |
| `INDUS_EPOCHS` | 80 | `src/train.py:735` | Training epochs |
| `INDUS_BATCH` | 16 | `src/train.py:736` | Batch size |
| `TF_CPP_MIN_LOG_LEVEL` | — | many modules | Defaults to `"3"` to silence TF logging |

### 7.2 In-code constants worth tuning

| Constant | Location | Value | Meaning |
|---|---|---|---|
| `EXPECTED_MATCH_MAP` | `src/evaluate.py:40`, `src/audit_validity.py:53` | 4 entries | Keeladi folder → acceptable Indus classes |
| `REJECTION_PREFIX` | `src/evaluate.py:57`, `src/train.py:76` | `"zz_"` | "Not an Indus sign" class prefix |
| Match threshold | `run_pipeline.py:91` | `0.5` | Lower = more candidate matches |
| `ZOOM` | `src/digitise_fig65.py:42` | `6.0` | PDF render resolution |
| `FIG65_ROWS` | `src/digitise_fig65.py:46` | 39 entries | serial → NFM / M-1977 / W-2015 / P-2010 |
| `STROKE_SERIALS` | `src/digitise_fig65.py:75` | 9 serials | Degenerate shape-match glyphs, excluded from claims |
| `ALLOGRAPH_GROUPS` | `src/train.py:48` | 4 groups | Fig.65 "or" rows merged into one sign |
| `ANNEXURE_REMAP` | `src/train.py:65` | 2 entries | M-1977-numbered images → true core class |
| `IMG` / `EMBED_DIM` / `N_CLASSES` | `src/siamese_embed.py:42-44` | 64 / 64 / 39 | Siamese input, embedding dim, class count |
| `KEELADI_NAMES` | `src/decoding.py:42` | 15 names | Attested names for the NLP layer |
| `CORPUS_READINGS` | `src/decoding.py:61` | 1 entry | Published attribution per inscription folder |
| `augment_factor` | `src/train.py:206` | 40 default / 25 via env | Augmentation copies per image |
| `epochs` | `src/train.py:631` | 80 | Default training epochs |

### 7.3 Generated artefacts

| Path | Written by |
|---|---|
| `models/indus_classifier.keras` + `indus_classifier_classes.txt` | `train.save_trained_model()` |
| `models/fig65_siamese.keras` | `siamese_embed.main()` |
| `models/evaluation_results/validity_audit.json` / `_report.txt` | `audit_validity.write_report()` |
| `models/evaluation_results/keeladi_evaluation_report.txt` | `evaluate.generate_report()` |
| `models/evaluation_results/keeladi_predictions.json` | `run_pipeline.write_predictions_json()` |
| `models/evaluation_results/sign_match_report.txt` / `.json` | `sign_matcher.write_report()` |
| `models/evaluation_results/siamese_report.json`, `fig65_embeddings.json` | `siamese_embed.main()` |
| `models/evaluation_results/known_pair_comparison.png`, `known_pair_scores.json` | `evaluate.generate_known_pair_comparison()` |
| `models/evaluation_results/match_statistics.png`, `confidence_distribution.png`, `confusion_matrix.png`, `training_history.png` | plotting methods |
| `models/evaluation_results/graffiti_gallery_*.png`, `graffiti_vs_indus_*.png` | gallery / comparison methods |
| `models/evaluation_results/decoded/decoded_readings.txt` | `decoding` via `evaluate.run_decoding()` |
| `data/results/logs/pipeline_log_*.txt` | `train.setup_logging()` |
| `data/results/logs/evaluation_log_*.txt` | `evaluate.setup_logging()` |
| `data/lexicon.json` | `InscriptionDecoder._load_lexicon()` (skeleton on first run) |

---
## 8. Extension Guide

| Goal | Where to change |
|---|---|
| Add a new sign class | Create `data/processed/train/primary_core_signs/sign_NN_PXXX/` and drop images in. Folder name becomes the label automatically |
| Change image size | Every module constructs `ImageNormalizer(target_size=...)` — change in one place per module |
| Add a CNN layer | `IndusClassifierCNN.build_model()` |
| Add a classification head | `IndusClassifierCNN.build_multi_head_model()` + a loss weight |
| Change augmentation | `IndusKeeladiTrainer._build_augmentation_pipeline()` and `_numpy_stochastic_degrade()` |
| Change the match threshold | `run_pipeline.py:91` (`threshold=0.5`) |
| Add an evaluation metric | `KeeladiEvaluator.analyze_civilization_link()` |
| Add a Brahmi letter | Add a folder under `data/processed/val/tamil_brahmi/general_brahmi_letters/` |
| Add an attested name | `KEELADI_NAMES` in `src/decoding.py` |
| Add a corpus reading | `CORPUS_READINGS` in `src/decoding.py` |
| Wire in grid decomposition | `GridDecomposer.extract_features_from_grid()` → concatenate to the CNN input or add a second head |
| Wire in the Siamese embedding | Embeddings are saved to `fig65_embeddings.json`; `src/sign_matcher.py` currently computes its own prototypes |

---

## 9. Design Decisions and Invariants

1. **Honesty gate.** `run_pipeline.py` and `process_model.py` never invent numbers. If the
   model or data is missing they stop and say so. Anything that cannot be computed is
   reported as `null`, never estimated. This replaced an earlier version that drew training
   curves from `np.random.normal`, confidences from `np.random.beta`, and printed a
   hard-coded "75.5% match rate".
2. **Split before augment.** The default `leakage_controlled` protocol splits originals
   first, so validation accuracy is not inflated by augmented clones of training images.
3. **One preprocessing path.** Train and val both go through `ImageNormalizer.process_image()`.
4. **Glyph bright on dark.** Enforced authoritatively by Otsu, because the corpus was built
   from two sources with mixed polarity.
5. **Autocrop before resize.** The glyph fills the 64×64 canvas regardless of source, which
   is the single largest accuracy gain in the preprocessing chain.
6. **Outline exclusion.** The pot outline is detected and reported as context, never
   classified, so it can never cause a false comparison.
7. **Rejection classes are not matches.** `zz_` predictions are excluded from match claims.
8. **Authored readings are labelled as such.** `cnn_potsherd_annotator` displays expected
   reference readings and says so in its docstring.

---
## 10. Known Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **Closed-set 43-way softmax** | Structurally cannot output "this is not an Indus sign". Every Keeladi sherd is forced onto some label, so "match rate" is not evidence on its own. Mitigated but not solved by `zz_` rejection classes and `sign_matcher.py` |
| 2 | **Training data is thin** | 980 images over 43 classes, largely one hand-drawn glyph plus augmented clones per class |
| 3 | **Fig.65 digitisation is incomplete** | 39 of 40 serials recovered; serial 40 is a photograph, not a drawing |
| 4 | **Stroke signs are not separable** | Serials 9–16 and 30 differ only in stroke count/position. Five score 0.000 top-1. They are flagged and excluded rather than down-weighted |
| 5 | **Siamese retrieval is an upper bound** | Test queries are perturbations of the training glyphs, not independent allographs |
| 6 | **Domain transfer not executed** | `WeightTransfer` is scaffolded but has never been run on real Keeladi data |
| 7 | **Grid decomposition not wired** | Features are implemented but never reach the model |
| 8 | **Corpus scale far below target** | 16 Keeladi validation images on disk, not the 1,001 sherds the research design targets |
| 9 | **Similarity ≠ descent** | The single biggest inferential risk. Visual proximity cannot establish script descent or decipherment |
| 10 | **Authored vs predicted readings** | Potentially confusing in annotator/decoder output; both modules state their provenance explicitly |

---

## 11. Environment Notes

- TensorFlow 2.21 on **native Windows runs CPU only** (Google disabled GPU mode from 2.11).
- For GPU: WSL2 + CUDA, or `tensorflow-directml-plugin`.
- Observed timing on an i7-class CPU: ~4 min / 20 epochs with 20× augmentation, so roughly
  16 min for 80 epochs.
- Run everything with the in-repo interpreter:
  `C:\Users\Administrator\Desktop\CNN\indus_keeladi_env\Scripts\python.exe <script>`

---

## 12. References

1. **The Indus Script — Recognition as an Alphabet** — 40-sign core alphabet, 3×3 grid
   decomposition, modifier + diacritic system (Figures 01–65).
2. **Keeladi Excavation Reports** — 1,001 graffiti sherds + 56 Tamil-Brahmi inscribed sherds,
   4-sign direct comparison table.
3. **Allographic Variety Table (Figure 59)** — style-invariant training targets.

Source PDFs are in `docs/`, with `*.extracted.txt` text extracts alongside.