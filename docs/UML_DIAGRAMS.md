# UML Diagrams — Indus–Keeladi CNN

> PlantUML sources for the Indus–Keeladi CNN project.
> Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md).
> The same sources also exist as standalone `.puml` files in `docs/uml/`.

## How to render

**VS Code** — install the *PlantUML* extension (jebbs), open a `.puml` file, press `Alt+D`.

**IntelliJ / JetBrains** — install *PlantUML Integration*.

**CLI (any OS)**

```bash
pip install plantuml
plantuml docs/uml/some_diagram.puml          # writes a .png next to the source
plantuml -tpdf docs/uml/some_diagram.puml    # or PDF
plantuml -tsvg  docs/uml/some_diagram.puml   # or SVG (best for print figures)
```

**Online** — <https://www.plantuml.com/plantuml/umlviewer> — paste and render.

## Index

| # | Diagram | File | Purpose |
|---|---|---|---|
| 1 | System Context (C4) | `01_system_context.puml` | Project boundary, actors, external systems |
| 2 | Component | `02_component.puml` | Modules and their dependencies |
| 3 | Class — Preprocessing & Model | `03_class_preprocessing_model.puml` | `ImageNormalizer`, `GridDecomposer`, `IndusClassifierCNN`, `WeightTransfer` |
| 4 | Class — Training & Evaluation | `04_class_training_evaluation.puml` | `IndusKeeladiTrainer`, `KeeladiEvaluator`, `InscriptionDecoder` |
| 5 | Class — Audit & Matching | `05_class_audit_matching.puml` | `audit_validity`, `sign_matcher`, `siamese_embed`, `digitise_fig65` |
| 6 | Sequence — Full Pipeline | `06_sequence_pipeline.puml` | `run_pipeline.main()` step by step |
| 7 | Sequence — Training | `07_sequence_training.puml` | Data load → augment → fit → save |
| 8 | Sequence — Evaluation | `08_sequence_evaluation.puml` | Sherd → predict → report |
| 9 | Activity — Preprocessing | `09_activity_preprocessing.puml` | `ImageNormalizer.process_image()` |
| 10 | Activity — Validity Audit | `10_activity_audit.puml` | The three audits |
| 11 | State — Model Lifecycle | `11_state_model_lifecycle.puml` | Untrained → trained → audited → reported |
| 12 | Deployment | `12_deployment.puml` | Runtime nodes and artefacts |
| 13 | Package / Namespace | `13_packages.puml` | Directory-to-namespace mapping |
| 14 | Use Case | `14_use_case.puml` | What each user-facing script lets you do |

---

## 1. System Context

```plantuml
@startuml
title Figure 1 — System Context (C4 Level 1)

actor Researcher as "Archaeologist / Researcher"
actor Reviewer as Reviewer

rectangle "Indus–Keeladi CNN\n(TensorFlow 2.21 / Python 3.11)" as System

database "data/processed/\n(train · val · lexicon)" as Data
database "models/\n(.keras · classes.txt)" as Models
folder "docs/*.pdf\n(source publications)" as PDFs
folder "models/evaluation_results/\n(reports · JSON · PNG)" as Reports

left to right direction

Researcher --> System : digitise glyphs,\ntrain, run pipeline,\nannotate sherds
Reviewer --> System : re-run audits,\nread reports
Reviewer --> Reports : verify every number

System --> Data : read images (folder = label)
System --> PDFs : extract Fig.65 glyphs (PyMuPDF)
System --> Models : read / write weights
System --> Reports : write reports (never fabricated)

note right of System
  Every reported number is computed
  from the model + images on disk.
  If data is missing the system
  STOPS rather than inventing results.
end note
@enduml
```

---

## 2. Component Diagram

```plantuml
﻿@startuml
title Figure 2 — Component Diagram

package "Entry points" {
  [run_pipeline.py ORCHESTRATOR] as RP
  [src/train.py] as TR
  [src/evaluate.py] as EV
  [src/audit_validity.py] as AV
  [src/sign_matcher.py] as SM
  [src/siamese_embed.py] as SE
  [src/digitise_fig65.py] as DG
  [process_model.py] as PM
  [dashboard.py] as DB
  [cnn_potsherd_annotator.py] as PA
}

package "Preprocessing" {
  [ImageNormalizer] as IN
  [GridDecomposer] as GD
}

package "Models" {
  [IndusClassifierCNN] as CNN
  [WeightTransfer] as WT
}

package "Analysis" {
  [InscriptionDecoder] as DEC
}

component "TensorFlow / Keras" as TF
component "OpenCV" as CV
component "scikit-learn" as SK
component "Matplotlib / Seaborn" as MP
component "PyMuPDF" as PMU
component "Streamlit" as SLT
component "data/lexicon.json" as LEX

RP --> AV
RP --> EV
RP --> SM
RP --> PA
RP ..> DB : writes

EV --> DEC
SM --> DEC
PM ..> AV : reads
DB ..> RP : reads

TR --> IN
TR --> CNN
EV --> IN
EV --> CNN
AV --> IN
SM --> IN
SE --> CV
DEC --> IN
DEC --> LEX

TR --> TF
EV --> TF
AV --> TF
SM --> TF
SE --> TF
CNN --> TF
WT --> TF

IN --> CV
DEC --> CV
SM --> CV
SE --> CV
GD --> CV
DG --> PMU
DG --> CV

EV --> SK
EV --> MP
DB --> MP
AV --> SK
TR --> SK

DB --> SLT
PA --> CV
@enduml
```
---

## 3. Class Diagram — Preprocessing & Model

```plantuml
@startuml
title Figure 3 — Preprocessing & Model Classes

class ImageNormalizer {
  - target_size : tuple
  - autocrop : bool
  - deskew : bool
  - morph_clean : bool
  + process_image(path, apply_threshold) : ndarray
  + process_array(arr, apply_threshold) : ndarray
  + process_directory(in_dir, out_dir) : int
  - load_image(path) : ndarray
  - convert_to_grayscale(img) : ndarray
  - _normalize_polarity(gray) : ndarray
  - _denoise(gray) : ndarray
  - _autocrop_to_glyph(gray) : ndarray
  - _deskew(gray) : ndarray
  - resize_image(img) : ndarray
  - normalize_pixel_values(img) : ndarray
  + apply_threshold(img, threshold) : ndarray
}

class GridDecomposer {
  - grid_rows : int
  - grid_cols : int
  + decompose_image(img) : list
  + analyze_grid_presence(cells, threshold) : ndarray
  + extract_features_from_grid(cells) : list
  + visualize_grid(img, presence, path) : ndarray
  - _calculate_horizontal_symmetry(cell) : float
  - _calculate_vertical_symmetry(cell) : float
}

class IndusClassifierCNN {
  - input_shape : tuple
  - num_classes : int
  - model : keras.Model
  + build_model() : Model
  + build_multi_head_model() : Model
  + train(X, y, X_val, y_val, epochs, batch_size) : History
  + predict(X) : ndarray
  + save_model(path) : void
  + load_model(path) : void
  + get_model_summary() : void
}

class WeightTransfer {
  - source_model : Model
  - target_model : Model
  + transfer_convolutional_layers() : int
  + freeze_early_layers(n) : void
  + unfreeze_all_layers() : void
  + progressive_unfreezing(epoch, total, schedule) : void
  + adapt_learning_rate(base_lr, factor) : Optimizer
  + domain_adaptation_training(X, y, epochs, batch_size) : History
  + evaluate_domain_gap(X_indus, X_keeladi) : dict
}

class "NormalizedGlyph\n<<value object>>" as NG

note right of NG
  64 x 64 x 1 float32 array
  glyph BRIGHT on dark background
  values in [0, 1]
end note

ImageNormalizer ..> NG : produces
IndusClassifierCNN ..> NG : consumes
GridDecomposer ..> NG : decomposes (not wired to CNN)
WeightTransfer ..> IndusClassifierCNN : copies Conv2D weights
@enduml
```

---

## 4. Class Diagram — Training & Evaluation

```plantuml
@startuml
title Figure 4 — Training & Evaluation Classes

class IndusKeeladiTrainer {
  - data_dir : Path
  - model_dir : Path
  - augment_factor : int
  - data_augmentation : keras.Sequential
  - classifier : IndusClassifierCNN
  + load_training_data(augment, split_protocol) : tuple
  + build_model(num_classes) : Model
  + train_model(X_tr, y_tr, X_va, y_va, epochs, batch_size) : History
  + save_trained_model(model_name) : void
  + load_trained_model(model_name) : void
  - _build_augmentation_pipeline() : Sequential
  - _numpy_stochastic_degrade(img) : ndarray
  - _augment_dataset(X, y) : tuple
  - _safe_train_val_split(X, y, test_size) : tuple
  - _safe_index_split(y, test_size, random_state) : tuple
}

class KeeladiEvaluator {
  - model_path : Path
  - data_dir : Path
  - class_names : list
  - normalizer : ImageNormalizer
  - classifier : IndusClassifierCNN
  + load_keeladi_validation_set() : dict
  + predict_keeladi_matches(data, threshold) : dict
  + analyze_civilization_link(predictions) : dict
  + generate_report(predictions, analysis, out_dir) : Path
  + run_decoding(out_dir) : Path
  + generate_known_pair_comparison(out_dir) : Path
  - _rejection_class_index() : int
  - _is_negative_folder(folder) : bool
  - _plot_match_statistics(analysis, out_dir) : void
  - _plot_confidence_distribution(predictions, out_dir) : void
  - _plot_graffiti_gallery(predictions, out_dir) : void
  - _save_single_sherd_comparisons(...) : void
}

class InscriptionDecoder {
  - data_dir : Path
  - normalizer : ImageNormalizer
  - classifier : IndusClassifierCNN
  - class_names : list
  - lexicon : dict
  - brahmi_refs : list
  + segment_inscription(path) : tuple
  + decode_inscription(path) : dict
  + decode_graffiti(img) : dict
  + match_brahmi(glyph) : dict
  + indus_top3(glyph) : list
  + nlp_decode(result, corpus_key) : dict
  + visualize_decoding(path, result, out_path) : void
  - _load_lexicon() : dict
  - _binarize(gray) : ndarray
  - _crop_to_bbox(binary, x, y, w, h) : ndarray
  - _recover_letters(mask, binary, h, w) : list
  - _projection_split(binary, x, y, w, h) : list
  - _load_brahmi_references() : list
}

enum SplitProtocol {
  leakage_controlled
  stratified
}

IndusKeeladiTrainer o-- SplitProtocol : uses
IndusKeeladiTrainer --> ImageNormalizer
IndusKeeladiTrainer --> IndusClassifierCNN
KeeladiEvaluator --> ImageNormalizer
KeeladiEvaluator --> IndusClassifierCNN
KeeladiEvaluator ..> InscriptionDecoder : run_decoding()
InscriptionDecoder --> ImageNormalizer
InscriptionDecoder --> IndusClassifierCNN : optional
InscriptionDecoder ..> lexicon : data/lexicon.json
@enduml
```

## 5. Class Diagram — Audit & Matching

```plantuml
@startuml
title Figure 5 — Audit, Retrieval & Digitisation

class "audit_validity\n<<module functions>>" as AV {
  + run_audit(model_path, data_dir, out_dir, n_perm) : dict
  + leakage_audit(X, y, groups, test_size, thresh, seed) : dict
  + openset_audit(model, class_names, normalizer, data_dir, n_pos, seed) : dict
  + verification_audit(model, class_names, normalizer, data_dir, n_perm, seed) : dict
  + write_report(audit, out_dir) : Path
  + load_train_corpus(data_dir, class_names, normalizer) : tuple
  + load_class_names(model_path) : list
  + load_classifier(model_path) : Model
  + predict_probs(model, images) : ndarray
  + expected_classes(folder) : list
  - _contamination(X, train_idx, val_idx, thresh) : dict
  - _negative_groups(normalizer, data_dir) : list
}

class "sign_matcher\n<<module functions>>" as SM {
  + match_all(model_path, data_dir, n_perm, seed) : dict
  + write_report(result, out_dir) : Path
  + binarize(gray) : ndarray
  + largest_component_mask(binary, size) : ndarray
  + stroke_coverage(a, b, dilate_iters) : float
  + embed_model(model) : Model
  + build_class_references(data_dir, class_names, limit) : dict
  + prototype_embeddings(embedder, refs) : ndarray
  + extract_glyphs(decoder, sherd_path) : list
  + score_sherd(glyphs, embedder, protos, refs, class_names) : dict
}

class "siamese_embed\n<<module functions>>" as SE {
  + load_glyphs() : list
  + fit_canvas(img, size, margin) : ndarray
  + augment(img, rng) : ndarray
  + build_dataset(items, per_class, seed) : tuple
  + build_test(items, seed) : tuple
  + build_reference(items) : ndarray
  + make_model() : Model
  + margin_triplet(emb, labels, margin) : scalar
  + train(items, epochs, per_class, batch, seed) : Model
  + retrieval_report(model, items, Xte, yte) : dict
  + per_class_report(model, items, Xte, yte) : dict
}

class "digitise_fig65\n<<module functions>>" as DG {
  + digitise(pdf, out_dir, zoom) : dict
  + table_rules(page) : tuple
  + clean_crop(img) : ndarray
  + contact_sheet(out, cell, cols) : Path
  + debug_page(pno) : void
  - _serial_anchors(page, serials) : dict
  - _words(page) : list
  - _bracket(values, lo, hi) : tuple
}

class AuditResult <<value object>>
class SherdScore <<value object>>
class "Fig65Glyph\n<<value object>>" as FG

note right of FG
  serial · nfm · m1977 · p2010
  is_stroke_sign
  (from manifest.json)
end note

AV --> ImageNormalizer : uses
AV --> FG : reads manifest
SM --> ImageNormalizer : uses
SM ..> InscriptionDecoder : segment_inscription()
SM ..> AuditResult : writes
SE --> FG : loads
SE ..> AuditResult : retrieval_report()
DG ..> FG : produces
AV ..> AuditResult : produces
@enduml
```

---

## 6. Sequence — Full Pipeline

```plantuml
@startuml
title Figure 6 — Sequence: run_pipeline.main()

actor User
participant RP as "run_pipeline.main()"
participant FS as "filesystem (PROJECT_ROOT)"
participant AV as "src.audit_validity"
participant EV as "src.evaluate.KeeladiEvaluator"
participant DEC as "src.decoding.InscriptionDecoder"
participant SM as "src.sign_matcher"
participant PA as cnn_potsherd_annotator

== Honesty gate ==
User -> RP : python run_pipeline.py
RP -> FS : MODEL_PATH.exists()?
alt model missing
  RP --> User : log.error + sys.exit(1)
  note right of RP: Refuses to fake results.
else model present
  == [1/6] Scan data ==
  RP -> FS : scan_data()
  FS --> RP : counts dict

  == [2/6] Validity audit ==
  RP -> AV : run_audit(model, data, EVAL_DIR)
  AV -> AV : leakage_audit(...)
  AV -> AV : openset_audit(...)
  AV -> AV : verification_audit(n_perm=2000)
  AV -> FS : validity_audit.json + _report.txt
  AV --> RP : audit dict

  == [3/6] Keeladi evaluation ==
  RP -> EV : KeeladiEvaluator(MODEL_PATH, DATA_DIR)
  EV -> EV : load_keeladi_validation_set()
  EV -> EV : predict_keeladi_matches(threshold=0.5)
  EV -> EV : analyze_civilization_link(predictions)
  EV -> FS : keeladi_evaluation_report.txt + PNGs
  EV -> DEC : run_decoding(EVAL_DIR)  [optional]
  DEC -> FS : decoded/decoded_readings.txt
EV --> RP : (predictions, analysis)

  == [4/6] Open-set sign matching ==
  RP -> SM : match_all(n_perm=2000)
  SM -> SM : segment, embed, stroke coverage
  SM -> SM : permutation test
  SM -> FS : sign_match_report.txt / .json
  SM --> RP : result

  == [5/6] Dashboard predictions ==
  RP -> FS : keeladi_predictions.json

  == [6/6] Annotate potsherds ==
  RP -> PA : PotsherdAnnotator(...).annotate_all()  [optional]
  PA --> RP : annotated images

  RP --> User : "PIPELINE COMPLETE" + paths + caveat
end
@enduml
```
---

## 7. Sequence — Training

```plantuml
@startuml
title Figure 7 — Sequence: src/train.py main()

actor User
participant T as "src.train.main()"
participant TR as IndusKeeladiTrainer
participant IN as ImageNormalizer
participant CNN as IndusClassifierCNN
participant FS as filesystem

User -> T : python src/train.py
T -> T : read INDUS_AUG=25, INDUS_EPOCHS=80, INDUS_BATCH=16
T -> TR : IndusKeeladiTrainer(data_dir, model_dir, augment_factor)

== Discover classes ==
TR -> FS : list primary_core_signs/ + indus_matched/
TR -> TR : canonical_class()\n[ALLOGRAPH_GROUPS + ANNEXURE_REMAP]
TR -> FS : fig65_real/manifest.json
TR -> TR : load_fig65_real_glyphs()  (mapped by SERIAL)
TR -> TR : generate_rejection_images()  zz_ families

== Preprocess originals ==
loop each image
  TR -> IN : process_image(path)
  IN -> IN : grayscale, denoise, Otsu polarity,\nautocrop, deskew, resize, [0,1]
  IN --> TR : 64x64x1 float32
end

== Leakage-controlled split ==
TR -> TR : _safe_index_split(y, test_size=0.2)
TR -> TR : SPLIT originals first
TR -> TR : _augment_dataset(X_train, y_train)\n(train split ONLY)

== Build + train ==
TR -> CNN : build_model(num_classes=43)
CNN --> TR : compiled Sequential model
TR -> TR : get_model_summary()
TR -> CNN : train(X_tr, y_tr, X_va, y_va, epochs, batch_size)
note right of CNN
  Adam 1e-3
  EarlyStopping(patience=10, restore_best_weights)
  ReduceLROnPlateau(factor=0.5, patience=5)
end note

== Save ==
TR -> FS : models/indus_classifier.keras
TR -> FS : models/indus_classifier_classes.txt
TR --> User : "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
@enduml
```

---

## 8. Sequence — Evaluation

```plantuml
@startuml
title Figure 8 — Sequence: KeeladiEvaluator

actor User
participant EV as KeeladiEvaluator
participant IN as ImageNormalizer
participant CNN as IndusClassifierCNN
participant DEC as InscriptionDecoder
participant FS as filesystem

User -> EV : load model + class names
EV -> FS : models/indus_classifier.keras
EV -> FS : models/indus_classifier_classes.txt
EV -> CNN : load_model(path)
EV --> User : "Loaded model with 43 classes"

== Load validation set ==
EV -> FS : walk data/processed/val/{keeladi,tamil_brahmi}
loop each image
  EV -> IN : process_image(path)
  IN --> EV : 64x64x1 array
end
EV --> User : validation_data dict

== Predict ==
loop each image
  EV -> CNN : predict(X)
  CNN --> EV : softmax (43,)
  EV -> EV : top-3 argsort
  EV -> EV : if argmax is zz_ prefix -> REJECT\n(never counted as match)
end

== Analyse + report ==
EV -> EV : analyze_civilization_link(predictions)
EV -> FS : keeladi_evaluation_report.txt
EV -> FS : match_statistics.png\nconfidence_distribution.png\ngraffiti_gallery_*.png
EV -> DEC : run_decoding(EVAL_DIR)
DEC -> FS : decoded/decoded_readings.txt
EV -> FS : known_pair_comparison.png\nknown_pair_scores.json
EV --> User : report path + summary
@enduml
```

---

## 9. Activity — Preprocessing

```plantuml
@startuml
title Figure 9 — Activity: ImageNormalizer.process_image()

start
:cv2.imread(path);
if (image is None?) then (yes)
  :raise ValueError;
  stop
endif
:convert_to_grayscale (BGR2GRAY);
:invert_if_dark_on_bright();
note right
  Legacy heuristic. Unreliable for
  sparse stroke glyphs - kept only
  for API compatibility.
end note
:_denoise() morphological open/close;
:_normalize_polarity() Otsu split;
note right
  Authoritative. Forces
  glyph BRIGHT on dark background.
end note
:_autocrop_to_glyph();
note right
  Contour bounding box + square pad.
  BIGGEST accuracy gain: the glyph
  fills the canvas regardless of source.
end note
:_deskew() by image moments;
if (apply_threshold?) then (yes)
  :apply_threshold(threshold=127);
endif
:resize to target_size (64x64) INTER_AREA;
:astype(float32) / 255.0;
:return array in [0, 1];
stop
@enduml
```

---

## 10. Activity — Validity Audit

```plantuml
@startuml
title Figure 10 — Activity: run_audit()

start
:run_audit(model_path, data_dir, out_dir, n_perm);
:load_class_names(model_path);
:load_classifier(model_path);

partition "1. Leakage audit" {
  :load_train_corpus()\n(mirrors src.train exactly);
  :leakage_audit(X, y, groups, thresh=0.99);
  partition "protocol A" {
    :random split (test_size=0.15);
  }
  partition "protocol B" {
    :source-disjoint (group) split;
  }
  :compute % val images with a >=0.99 twin;
  :emit verdict (clean / contaminated);
}

partition "2. Open-set audit" {
  :build negative groups\n(blank, saturated, noise, Brahmi letters);
  :predict_probs(model, negatives);
  :predict_probs(model, positives);
  :false-acceptance rate vs positive control;
}

partition "3. Verification audit" {
  :load the 4 curated Keeladi pairs;
  :predict expected vs actual top-1/top-3;
  :permutation test (n_perm=2000)\nfor better-than-chance;
}

:write_report() -> validity_audit.json + _report.txt;
:any value that cannot be computed is written as null;
stop
@enduml
```
---

## 11. State — Model Lifecycle

```plantuml
@startuml
title Figure 11 — State: Trained Model Lifecycle

[*] --> Untrained : repo clone

state Untrained {
  [*] --> waiting
  waiting : no .keras on disk
}

Untrained --> Digitising : python -m src.digitise_fig65
state Digitising {
  [*] --> rendering
  rendering : render Fig.65 at 6x\ncrop cells, clean rules
  rendering --> written : 39 glyphs + manifest.json
  written : fig65_real/*.png
}
Digitising --> Untrained

Untrained --> Training : python src/train.py
state Training {
  [*] --> splitting
  splitting : split originals (leakage_controlled)
  splitting --> augmenting : train split only
  augmenting --> fitting
  fitting : Adam 1e-3 + EarlyStopping\n+ ReduceLROnPlateau
  fitting --> saved
  saved : indus_classifier.keras + classes.txt
}
Training --> Trained

state Trained {
  [*] --> ready
  ready : runnable by evaluate / audit / matcher
}

Trained --> Evaluating : python run_pipeline.py
state Evaluating {
  [*] --> auditing
  auditing : leakage, open-set, verification
  auditing --> scoring
  scoring : predict_keeladi_matches(threshold=0.5)
  scoring --> matching
  matching : sign_matcher.match_all()
  matching --> reported
  reported : reports + JSON + PNGs written
}
Evaluating --> Reported

state Reported {
  [*] --> available
  available : process_model.py reads it\ndashboard.py displays it
}

Reported --> Evaluating : re-run after new data

note right of Evaluating
  If the model is missing the pipeline
  exits with an error instead of
  generating placeholder metrics.
end note

state "Rejected" as Rejected
note left of Rejected
  A prediction landing on a zz_*
  class is reported as REJECT,
  never as a match.
end note
Evaluating --> Rejected : top-1 == zz_*
@enduml
```

---

## 12. Deployment Diagram

```plantuml
@startuml
title Figure 12 — Deployment (Windows workstation, CPU-only TF)

node "Developer workstation\nWindows · i7-class CPU" {

  node "Virtual environment\nindus_keeladi_env/ (Python 3.11)" {
    artifact "TensorFlow 2.21 / Keras 3\nCPU only on native Windows" as TF
    artifact "OpenCV + NumPy" as CV
    artifact "scikit-learn" as SK
    artifact "Matplotlib + Seaborn" as MP
    artifact "PyMuPDF" as PMU
  }

  folder "data/processed/\ntrain · val · test" as D1
  folder "data/lexicon.json" as D2
  folder "data/results/logs/" as D3
  folder "docs/*.pdf" as D4
  folder "models/*.keras + classes.txt" as M1
  folder "models/evaluation_results/" as M2
  folder "docs/figures/" as D5

  artifact "indus_keeladi_env\\Scripts\\python.exe" as PY
  artifact "indus_keeladi_env\\Scripts\\streamlit.exe" as ST

  PY --> TF
  PY --> CV
  PY --> SK
  PY --> MP
  PY --> PMU
  ST --> MP

  PY ..> D1 : reads
  PY ..> D2 : reads / writes
  PY ..> D3 : writes
  PY ..> D4 : reads (Fig.65)
  PY ..> M1 : reads / writes
  PY ..> M2 : writes
  PY ..> D5 : writes contact sheet
  ST ..> M2 : reads
}

note bottom of TF
  GPU is unavailable on native Windows
  since TensorFlow 2.11. For GPU use
  WSL2 + CUDA or tensorflow-directml-plugin.
  Observed CPU timing: ~4 min / 20 epochs
  at 20x augmentation.
end note
@enduml
```

## 13. Package / Namespace Diagram

```plantuml
@startuml
title Figure 13 — Packages = Directories

package "CNN (project root)" {

  package "src.preprocessing" {
    [image_normalization .ImageNormalizer]
    [grid_decomposition .GridDecomposer]
  }

  package "src.models" {
    [indus_classifier_cnn .IndusClassifierCNN]
    [weight_transfer .WeightTransfer]
  }

  package "src (analysis)" {
    [train .IndusKeeladiTrainer]
    [evaluate .KeeladiEvaluator]
    [audit_validity .run_audit]
    [sign_matcher .match_all]
    [siamese_embed .main]
    [digitise_fig65 .digitise]
    [decoding .InscriptionDecoder]
  }

  package "presentation" {
    [dashboard]
    [cnn_potsherd_annotator .PotsherdAnnotator]
    [cnn_annotation_generator .CNNAnnotationGenerator]
    [fullscreen_resizer .FullscreenResizer]
    [secret_resizer]
    [process_model]
    [reorganize_brahmi]
  }

  package "orchestration" {
    [run_pipeline (orchestrator)]
  }

  package "notebooks" {
    [dataset_analysis.ipynb]
    [validation_report.ipynb]
  }

  package "docs" {
    [ANALYSIS_ACCURACY_ROADMAP.md]
    [ARCHITECTURE.md]
    [UML_DIAGRAMS.md]
    [uml/*.puml]
  }

  [run_pipeline] ..> [src.analysis] : imports
  [src.analysis] ..> [src.preprocessing] : imports
  [src.analysis] ..> [src.models] : imports
  [sign_matcher] ..> [decoding] : reuse segmentation
  [sign_matcher] ..> [audit_validity] : reuse helpers
  [evaluate] ..> [decoding] : run_decoding()
  [audit_validity] ..> [train] : reuses canonical_class,\nALLOGRAPH_GROUPS, ANNEXURE_REMAP
  [presentation] ..> [src.analysis] : reads artefacts
}
@enduml
```

---

## 14. Use Case Diagram

```plantuml
@startuml
title Figure 14 — Use Cases

actor Researcher
actor Reviewer

rectangle "Indus–Keeladi CNN" {
  usecase "UC1 Digitise Fig.65 glyphs\n(python -m src.digitise_fig65)" as UC1
  usecase "UC2 Train classifier\n(python src/train.py)" as UC2
  usecase "UC3 Run full pipeline\n(python run_pipeline.py)" as UC3
  usecase "UC4 Evaluate only\n(python src/evaluate.py)" as UC4
  usecase "UC5 Validity audit\n(python -m src.audit_validity)" as UC5
  usecase "UC6 Open-set sign match\n(python -m src.sign_matcher)" as UC6
  usecase "UC7 Metric learning\n(python -m src.siamese_embed)" as UC7
  usecase "UC8 Read results\n(python process_model.py)" as UC8
  usecase "UC9 Interactive dashboard\n(streamlit run dashboard.py)" as UC9
  usecase "UC10 Annotate potsherds\n(PotsherdAnnotator)" as UC10
  usecase "UC11 Tune annotation boxes\n(fullscreen_resizer.py)" as UC11
  usecase "UC12 EDA / publication plots\n(notebooks/*.ipynb)" as UC12

  usecase "Add a new sign class\n(drop images into a class folder)" as UC13
  usecase "Change epochs / augmentation\n(INDUS_EPOCHS, INDUS_AUG, INDUS_BATCH)" as UC14
  usecase "Transfer to Keeladi domain\n(WeightTransfer)" as UC15 <<scaffolded>>
}

Researcher --> UC1
Researcher --> UC2
Researcher --> UC3
Researcher --> UC4
Researcher --> UC10
Researcher --> UC11
Researcher --> UC13
Researcher --> UC14

Reviewer --> UC3
Reviewer --> UC5
Reviewer --> UC6
Reviewer --> UC7
Reviewer --> UC8
Reviewer --> UC12

UC3 ..> UC5 : <<includes>>
UC3 ..> UC4 : <<includes>>
UC3 ..> UC6 : <<includes>>
UC3 ..> UC10 : <<includes>>
UC4 ..> UC1 : <<requires>> real glyphs
UC2 ..> UC1 : <<uses>>
UC8 ..> UC3 : <<requires>>
UC9 ..> UC3 : <<requires>>
UC6 ..> UC4 : <<requires>> segmentation
UC15 ..> UC2 : <<extends>>
@enduml
```

---

## Diagram Conventions

| Convention | Meaning |
|---|---|
| `-->` | Direct call / data flow |
| `..>` | Dependency, import, or "uses/includes" |
| `o--` | Aggregation |
| `<<value object>>` | Immutable data holder |
| `<<scaffolded>>` | Implemented but never executed on real data |
| `#` prefix | Private member |
| `+` prefix | Public member |

## Validating the sources

Render them all at once:

```powershell
python -m pip install plantuml
Get-ChildItem docs\uml\*.puml | ForEach-Object { plantuml -tsvg $_.FullName }
```

Any syntax error is reported on stderr with the offending line number.
  PY --> PMU
  ST --> MP

  PY ..> D1 : reads
  PY ..> D2 : reads / writes
  PY ..> D3 : writes
  PY ..> D4 : reads (Fig.65)
  PY ..> M1 : reads / writes
  PY ..> M2 : writes
  PY ..> D5 : writes contact sheet
  ST ..> M2 : reads
}

note bottom of TF
  GPU is unavailable on native Windows
  since TensorFlow 2.11. For GPU use
  WSL2 + CUDA or tensorflow-directml-plugin.
  Observed CPU timing: ~4 min / 20 epochs
  at 20x augmentation.
end note
@enduml
```
  :leakage_audit(X, y, groups, thresh=0.99);
  partition "protocol A" {
    :random split (test_size=0.15);
  }
  partition "protocol B" {
    :source-disjoint (group) split;
  }
  :compute % val images with a >=0.99 twin;
  :emit verdict (clean / contaminated);
}

partition "2. Open-set audit" {
  :build negative groups\n(blank, saturated, noise, Brahmi letters);
  :predict_probs(model, negatives);
  :predict_probs(model, positives);
  :false-acceptance rate vs positive control;
}

partition "3. Verification audit" {
  :load the 4 curated Keeladi pairs;
  :predict expected vs actual top-1/top-3;
  :permutation test (n_perm=2000) for better-than-chance;
}

:write_report() -> validity_audit.json + _report.txt;
:any value that cannot be computed is written as null;
stop
@enduml
```
  EV --> RP : (predictions, analysis)

  == [4/6] Open-set sign matching ==
  RP -> SM : match_all(n_perm=2000)
  SM -> SM : segment, embed, stroke coverage
  SM -> SM : permutation test
  SM -> FS : sign_match_report.txt / .json
  SM --> RP : result

  == [5/6] Dashboard predictions ==
  RP -> FS : keeladi_predictions.json

  == [6/6] Annotate potsherds ==
  RP -> PA : PotsherdAnnotator(...).annotate_all()  [optional]
  PA --> RP : annotated images

  RP --> User : "PIPELINE COMPLETE" + paths + caveat
end
@enduml
```
  + predict_keeladi_matches(data, threshold) : dict
  + analyze_civilization_link(predictions) : dict
  + generate_report(predictions, analysis, out_dir) : Path
  + run_decoding(out_dir) : Path
  + generate_known_pair_comparison(out_dir) : Path
  - _rejection_class_index() : int
  - _is_negative_folder(folder) : bool
  - _plot_match_statistics(analysis, out_dir) : void
  - _plot_confidence_distribution(predictions, out_dir) : void
  - _plot_graffiti_gallery(predictions, out_dir) : void
  - _save_single_sherd_comparisons(...) : void
}

class InscriptionDecoder {
  - data_dir : Path
  - normalizer : ImageNormalizer
  - classifier : IndusClassifierCNN
  - class_names : list
  - lexicon : dict
  - brahmi_refs : list
  + segment_inscription(path) : tuple
  + decode_inscription(path) : dict
  + decode_graffiti(img) : dict
  + match_brahmi(glyph) : dict
  + indus_top3(glyph) : list
  + nlp_decode(result, corpus_key) : dict
  + visualize_decoding(path, result, out_path) : void
  - _load_lexicon() : dict
  - _binarize(gray) : ndarray
  - _crop_to_bbox(binary, x, y, w, h) : ndarray
  - _recover_letters(mask, binary, h, w) : list
  - _projection_split(binary, x, y, w, h) : list
  - _load_brahmi_references() : list
}

enum SplitProtocol {
  leakage_controlled
  stratified
}

IndusKeeladiTrainer o-- SplitProtocol : uses
IndusKeeladiTrainer --> ImageNormalizer
IndusKeeladiTrainer --> IndusClassifierCNN
KeeladiEvaluator --> ImageNormalizer
KeeladiEvaluator --> IndusClassifierCNN
KeeladiEvaluator ..> InscriptionDecoder : run_decoding()
InscriptionDecoder --> ImageNormalizer
InscriptionDecoder --> IndusClassifierCNN : optional
InscriptionDecoder ..> lexicon : data/lexicon.json
@enduml
```