# Indus–Keeladi CNN — Deep Analysis, Accuracy Roadmap, Problem Statement & Research Gap

> Scope: audit of the code in this repository + a research-grounded plan to reach *credible* accuracy.
> Author: automated code + literature review. All line references point at the files as they exist in this workspace.

---

## 0. TL;DR (read this first)

1. The repository is a **well-structured deep-learning scaffold** (train → evaluate → decode → dashboard) for a genuinely interesting hypothesis: that Indus signs (~2600–1900 BCE) → Keeladi graffiti (~6th c. BCE) → Tamil-Brahmi are the same evolving sign tradition.
2. **The headline numbers in the repo are not produced by the model.** `run_pipeline.py`, `process_model.py`, and `dashboard.py` *fabricate* the training curves, confidences and match scores with `np.random.*` and hard-coded constants. The `keeladi_evaluation_report.txt` even contradicts itself (says "Match Rate 18.8%" in the header and "75.5% match rate" in the narrative).
3. The trained CNN itself is real (`models/indus_classifier.keras`), **but** it is trained on **self-generated synthetic glyphs** (1 drawing → 20 augmented copies per class) and validated on **augmented copies of the same glyphs** — a textbook leak that inflates accuracy to the 90–97% range while meaning very little.
4. The modelling task is framed as a **closed-set 49-way softmax**, which structurally *cannot* say "this sherd is not an Indus sign". Every Keeladi sherd is forced onto some Indus label, so "match rate" is not evidence of anything.
5. Even with a perfect CNN, **visual/activation similarity is not linguistic descent.** That inferential leap is the single biggest "invalid analysis" risk in the whole project.
6. The Google Notebook link you sent (`notebook.google.com/...`) is a **private NotebookLM share** that requires Google sign-in — I could not read it. Paste the text or make it public and I'll fold it in.

The rest of this document gives the evidence, the general failure taxonomy, the expanded problem statement, the research gap, the civilization "connecting link" with real reference papers, and a concrete roadmap to defensible accuracy.

---

## 1. What the project actually is

Goal (from `README.md`): use a CNN to quantify the *evolutionary* link between the **Indus Valley Script** and **Keeladi graffiti**, bridging a ~2,000-year gap, and extend it to **Tamil-Brahmi**.

Pipeline (code map):

| Stage | File | What it does |
|---|---|---|
| Preprocess | `src/preprocessing/image_normalization.py` | grayscale → polarity invert → denoise → **autocrop to glyph contour** → **deskew by image moments** → 64×64 → [0,1] |
| Decompose | `src/preprocessing/grid_decomposition.py` | 3×3 cell split, per-cell density + H/V symmetry features |
| Model | `src/models/indus_classifier_cnn.py` | VGG-style 6-conv CNN, `num_classes` softmax; also a 3-head variant (sign/modifier/diacritic) |
| Transfer | `src/models/weight_transfer.py` | progressive unfreeze + domain-adaptation fine-tune + domain-gap metric |
| Train | `src/train.py` | loads folder-per-class data, augments, splits 85/15, 80 epochs |
| Evaluate | `src/evaluate.py` | runs every sherd through the model, top-3 per sherd, galleries, confusion matrix |
| Decode | `src/decoding.py` | segments a sherd into letters, dual-reads Brahmi **template match** + Indus **CNN**, NLP composes a word |
| Report | `run_pipeline.py` / `process_model.py` | writes `keeladi_evaluation_report.txt`, plots, `known_pair_scores.json` |
| UI | `dashboard.py` | Streamlit viewer (stats / matching / decoder) |

Dataset on disk (counted): **45 primary core signs × 20 images**, 5 matched signs × 20, diacritics ~1 each, **12** general Keeladi graffiti, **4** matched Keeladi (1 each), **21** Tamil-Brahmi inscriptions, **36** Brahmi letters. The "20 images" per class are `aug_00…aug_19` — i.e. an **offline augmentation of a single source glyph**.

---

## 2. Invalid analysis already present in THIS codebase (with evidence)

These are not hypothetical — they are in the shipped code and are why the reported results cannot be trusted.

### 2.1 Fabricated metrics (the most serious issue)

`run_pipeline.py` — the script whose banner says "COMPLETE PIPELINE":

- L108: comment literally says `# Simulate model training with realistic metrics`
- L122–126: training curves are pure noise-sculpted maths:
  ```python
  train_loss = 2.0 - (epochs / 80) * 1.6 + np.random.normal(0, 0.04, 80)
  val_acc    = (epochs / 80) * 0.92 + np.random.normal(0, 0.025, 80)
  ```
- L148: confidences are `np.random.beta(8, 2, size=num_images)`
- L153–155: predicted classes are `np.random.choice(train_classes, size=3)` and scores `np.random.dirichlet(...)`
- L326–328: Precision/Recall are `accuracy * 0.96 / 0.98 "estimated"`, and **F1 = 2·acc·acc/(acc+acc) = acc** (algebraically wrong)
- L361–382: the "Computational evidence" narrative is a **hard-coded string** that states a "75.5% match rate" and "High confidence matches (>80%) observed in all 4 key Keeladi samples" — contradicted by the same script's own computed header (`Match Rate: 18.8%`, `High Confidence Matches: 3/16`).
- L399: `known_pair_scores.json` entries are `np.random.uniform(75, 95)`.

`process_model.py`:
- L46: `# Generate dummy but realistic metrics`; `accuracy: 0.92, match_rate: 75.5, high_conf_matches: int(total * 0.60)`
- L141: `confidence_scores = np.random.beta(8, 2, size=100)`
- L166–167: known-pair scores `np.random.uniform(75, 95)`.

`dashboard.py`:
- L158–159: defaults `match_pct = 85.0`, `confidence = 0.80` if the JSON is missing.
- L246–275: `predictions_map` (top-5 "CNN predictions" per sign) is **hard-coded**, despite the label "derived from real classifier output".

**Consequence:** every number a reviewer would cite (accuracy, match rate, confidence, top-5) can be produced on a fresh clone with zero training. This is the definition of an invalid result.

### 2.2 Data leakage — validation is contaminated

`src/train.py` L288–293:

```python
# Augment first, then split (so augmented versions stay with their originals in train/val)
if augment:
    X, y = self._augment_dataset(X, y)
X_train, X_val, y_train, y_val = self._safe_train_val_split(X, y, test_size=0.15)
```

The comment states the bug as if it were a feature. Because `_augment_dataset` (L141–173) emits the original **and** 25–40 augmented clones *before* splitting, near-identical images of the same glyph land in both train and val. Validation accuracy therefore measures "can I recognise a slightly perturbed copy of something I've already seen", not generalisation. This is exactly why the README can claim "val accuracy ≥ 97%".

### 2.3 Synthetic, self-referential training data

Each `sign_XX_*` folder contains one drawing and its 19–39 augmented clones (`aug_00.png` …). The original `generate_sample_data.py` (now deleted) auto-expanded a single glyph into "20 realistic variants". So the CNN is learning a distribution **it defined itself**, on matplotlib-like clean line art — not on real archaeological scans. Training on synthetic data is legitimate *only* if at least the test set is real; here it is not.

### 2.4 Closed-set softmax cannot express "unknown"

`evaluate.py` L30–39 defines only 4 expected matches and a rejection class prefix `"zz_"`, but the shipped `indus_classifier_classes.txt` has **no `zz_` class** — all 49 classes are Indus signs. So `predict_keeladi_matches()` must assign every sherd (including a broken pot rim or a random scratch) to one of 49 signs. With no negatives, **false-positive matches are guaranteed by construction**, and "match rate" partly measures the threshold, not the archaeology.

### 2.5 Hard-coded "decodings" masquerading as model output

- `cnn_annotation_generator.py` L17–67: `ATAN_DATA` hard-codes each sherd's reading (`ma-ta-na`, `ka-ma-ra`, …) and an `indus` sign id with a fixed `confidence` — these are *authored*, not inferred.
- `cnn_potsherd_annotator.py` L151+ `_load_character_mappings()` returns fixed `(brahmi, indus, conf)` tuples per slot.
- The green `I:P###` label drawn on the image is therefore a **look-up**, while the code path presents it as a CNN match.

### 2.6 Wrong / misleading statistics

- F1 formula is degenerate (§2.1).
- "Precision/Recall (estimated)" are not measured.
- No cross-validation, no confidence intervals, no significance test — yet the report says "mathematical proof" (L380–381). With n=4 matched sherds and n=16 analysed, no such claim is statistically supportable.

### 2.7 The inferential fallacy (independent of the code)

"Indus → Keeladi → Tamil-Brahmi" is asserted as a chain. But:
- **Tamil-Brahmi descends from Brahmi, not from Indus** — this is settled epigraphy. Keeladi graffiti being *precursors* of Brahmi is a *hypothesis*, not a proof, and the Keeladi report itself calls it "an initial finding" (conversation.txt / TN Dept. of Archaeology 2019).
- **Shape similarity across scripts is common and often coincidental.** Small alphabets of straight strokes and crosses (X, +, ⊃, |, ○) emerge independently worldwide. A CNN firing on "cross" is *not* evidence the two systems share ancestry — the repo's own cross-match (P225) is the clearest example of a shape that is near-universal.

---

## 3. The general class of "invalid analysis" CNN projects usually have

These are the recurring failure modes of CNN-based script/sign papers — and every one of them is present or latent here. Use this as a checklist when writing your methodology chapter.

**A. Evaluation-protocol failures**
1. **Train/val leakage** via augmenting before splitting (§2.2).
2. **Test-set reuse** — class names, hyper-parameters and thresholds tuned on the same data used to report accuracy.
3. **Offline augmentation of a single source image** counted as independent samples.
4. **No held-out real test set** — synthetic train *and* synthetic test.
5. **Class imbalance ignored** — accuracy dominated by the 45 big classes; macro-F1/balanced accuracy never reported.

**B. Statistical failures**
6. **No cross-validation / no confidence intervals / no error bars.** A single 85/15 split on ~900 images gives ±several points of noise.
7. **Softmax scores reported as probabilities of truth.** Uncalibrated CNNs are routinely over-confident; a 0.9 softmax ≠ 90% correctness.
8. **Tiny-n conclusions** — "supports the hypothesis" from 4–16 examples with no significance test.
9. **p-hacking by threshold** — match rate depends on the arbitrary 0.5/0.8 cut, never validated.

**C. Problem-framing failures (the deepest ones)**
10. **Closed-set classification for an open-set question.** Forcing every sherd into a known alphabet manufactures matches (§2.4). Correct framing is *open-set recognition* / *verification* (same/different), not 49-way softmax.
11. **Confusing correlation of shape with causation of descent.** A CNN is a shape-similarity machine; it has no notion of genealogy, phonology or chronology. Activation overlap is evidence of *visual resemblance*, which is necessary but nowhere near sufficient for a *script-descent* claim.
12. **Ignoring allographic variation vs. distinct signs.** Same sign, many hands/engraving styles (allographs) inflate apparent "matches"; different signs sharing a component deflate them. This is exactly what `grid_decomposition.py` was meant to help with but is never fed to the model.
13. **Domain shift treated as noise.** Clean synthetic line-art vs. photographed, occluded, low-contrast sherds is a *domain adaptation* problem, not a "just augment harder" problem.
14. **No negative controls.** There is no class of known *unrelated* marks to prove the model can *reject*. Without a control, a high match rate is unfalsifiable.

**D. Reproducibility failures**
15. **Fabricated/simulated outputs** mixed in with real ones (§2.1) — the worst possible kind, because it destroys trust in the *real* parts too.
16. **Hard-coded labels/results** in the visualisation layer (§2.5).
17. **No seed, no environment lock, no data checksum.**
18. **No baseline.** Nothing to compare the CNN against (e.g. shape descriptors, template matching, human expert agreement).

---

## 4. Problem statement (expanded)

**Domain context.** The Indus Valley Script (IVS), c. 2600–1900 BCE, is the largest corpus of the ancient world that remains undeciphered: ~4,000–5,000 inscribed objects, ~400–600 distinct signs, short texts (avg. ~5 signs), no bilingual key, and no confirmed language. Meanwhile, the Tamil Nadu State Department of Archaeology's Keeladi (Keezhadi) excavations (2015–) report ~**1,001 graffiti marks** on sherds dated to the **6th century BCE**, plus **56 Tamil-Brahmi** inscribed sherds. The excavators' 2019 report hypothesises that these graffiti are the "**missing link**" between the Indus script and Brahmi/Tamil-Brahmi, and illustrate the claim with **4 manually matched signs** (reported as P225, P307, P318, P365 in the sources used here).

**The core research problem.**

> *Given (a) an undeciphered script with no labelled vocabulary and only shape-level evidence, (b) a 2,000-year chronological gap, and (c) only a handful of hand-picked visual correspondences, can a computational method produce a **scalable, objective, reproducible and statistically defensible** measure of sign-level relatedness across the Indus → Keeladi-graffiti → Tamil-Brahmi sequence — and can that measure be validated so that "looks like" is not mistaken for "descends from"?*

**Sub-problems.**
1. **Digitisation & normalisation** — turn heterogeneous archaeological images (photos, scans, line drawings, screenshots) into a comparable representation (crop, deskew, polarity, resolution) *for both train and test* without leaking source identity.
2. **Recognition under open-set conditions** — identify signs among ~400+ IVS classes *and* correctly reject marks that belong to none (fragments, decoration, damage).
3. **Allograph modelling** — separate *the same sign written differently* from *different signs that share components*.
4. **Cross-domain matching** — compare across a severe domain gap (engraved, abraded sherd photos vs. catalogue drawings) and a severe *style* gap (thick catalogue strokes vs. thin scratches).
5. **Inference discipline** — report what a shape-similarity model can legitimately support (visual correspondence + priors) and explicitly refrain from claiming decipherment/descent that the evidence cannot bear.
6. **Reproducibility** — no fabricated metrics; every reported number traceable to model output on a held-out set.

**What a valid deliverable must report (success criteria).**
- Held-out, source-disjoint test accuracy **and** macro-F1, with cross-validated CIs.
- An explicit **open-set** metric (AUROC of "is this an Indus sign at all?") and a rejection rate on known non-Indus negatives.
- A **calibrated** confidence (reliability diagram / ECE), not raw softmax.
- An **ablation** vs. a non-deep baseline (HOG/SIFT/Zernike + SVM, and template matching), plus human-expert agreement.
- Statistical test (e.g. permutation/bootstrap) for any "more similar than chance" claim.
- Full reproducibility: seeds, versions, data hashes.

---

## 5. Research gap

The `README.md` already names four gaps (Scale, Subjectivity, Transformation, Decomposition). Here they are, sharpened, plus the gaps the project *misses*.

**Gap 1 — Scale.** Existing claims rest on ~4 hand-matched signs against 1,001 sherds and a ~400+ sign IVS inventory. Manual comparison cannot scale; there is no automated pipeline that scores every sherd against the whole alphabet.

**Gap 2 — Subjectivity.** "Resemblance" is judged by eye, without a reproducible, quantitative, pre-registered criterion. No published work gives a probability distribution per sherd with a validated decision threshold.

**Gap 3 — Transformation / chronology.** Nobody has mapped *feature-level* evolution across the three stages (IVS → graffiti → Brahmi) with a method that separates **stylistic allographs** from **actual sign change over time**.

**Gap 4 — Decomposition.** Compound ligatures (a core sign + modifiers + diacritics) are treated as atomic images, so the CNN cannot generalise composition, even though the literature describes a component system (the project's 3×3 grid idea).

**Gap 5 — Open-set / negative controls (MISSING in the repo).** No method here (or commonly published) proves the system can *say no*. Without negatives and an OOD score, match rates are unfalsifiable.

**Gap 6 — Domain adaptation (MISSING).** The synthetic→archaeological gap is not addressed by principled domain adaptation (only by ad-hoc augmentation).

**Gap 7 — Statistical rigour (MISSING).** No CIs, no significance testing, no expert-agreement baseline. "Mathematical proof" is claimed where only a point estimate exists.

**Gap 8 — Explainability tied to evidence (PARTIAL).** Grad-CAM/saliency exists conceptually but is not used to show *which strokes* drive a match, nor is it reconciled with the archaeological feature vocabulary.

**Gap 9 — Reproducibility (MISSING).** No seed/version manifest, and — critically — fabricated outputs. The literature's computational-epigraphy papers rarely release full, runnable reproducibility packages; this project is positioned to *lead* here if the fakes are removed.

**Positioning statement for your paper.** *"Prior computational work on Indus/Brahmi relatedness is limited by manual, small-n, closed-set comparison. We present the first reproducible, open-set, statistically-validated pipeline that (i) scores the full Keeladi corpus against the IVS inventory, (ii) explicitly rejects non-signs, (iii) separates allographs from distinct signs, and (iv) reports only model-derived, calibrated results with negative controls."*

---

## 6. The "connecting link between civilizations" + reference research

### 6.1 What the link actually is

The "connecting link" thesis has three legs, which must be kept distinct (conflating them is a common error):

| Leg | Claim | Status |
|---|---|---|
| **Chronological bridge** | De-urbanisation after 1900 BCE leaves a gap; "graffiti marks" on pottery persist from Late Harappan through the South Indian Iron Age to the 6th c. BCE. | Graffiti marks clearly exist; that they form an *unbroken written* tradition is **hypothesised**. |
| **Palaeographic (shape) bridge** | A handful of Keeladi graffiti shapes resemble IVS signs (the repo's P225/P307/P318/P365). | Observed but **small-n and confounded by universal shapes**. |
| **Linguistic/decipherment bridge** | The IVS language is Dravidian, so IVS → graffiti → Tamil-Brahmi is one lineage. | **Unproven.** Dravidian (Mahadevan, Parpola) is the leading hypothesis but not established; aDNA (Shinde et al. 2019) shows Indus people lacked Steppe ancestry — *consistent with* but not *proof of* a Dravidian language. |

Honest framing for a paper: **"we test the palaeographic leg computationally and quantitatively, and we do not claim the other two."**

### 6.2 The specific claim under test (Keeladi)

- TN State Dept. of Archaeology, **Keeladi Excavation Report (2019)**: announces the Indus–Tamil-Brahmi "link", ~1,001 graffiti marks, 56 Tamil-Brahmi sherds, 4 illustrated matches; explicitly calls it "an initial finding".
- Press: *The News Minute*, 19 Sep 2019 — Commissioner T. Udhayachandran: *"It's an initial finding… We found 1000 different marks. We have chosen a few that distinctly relate to the Indus."* (This is the exact gap the project's "Scale" claim targets.)
- **Dec 2025**: the TN government again announced a high-profile IVS↔Tamil (Brahmi) linkage effort — good as a *motivation* citation; treat official statements as hypotheses, not results.

### 6.3 Annotated reference list (three buckets)

**(A) The link / ancestry debate (read these to state your problem correctly)**
1. **Mahadevan, I.** *The Indus Script: Texts, Concordance and Tables* (1977); *Early Tamil Epigraphy* (2003) — standard sign catalogue + Dravidian framework.
2. **Parpola, A.** *Deciphering the Indus Script* (1994); *The Roots of Hinduism* (2015) — strongest Dravidian-rebus argument; also the component/modifier system your `grid_decomposition.py` echoes.
3. **Rao, R. P. N., et al.** "Entropic Evidence for Linguistic Structure in the Indus Script," *Science* 324 (2009) — statistical claim IVS is language-like.
4. **Farmer, S., Sproat, R., Witzel, M.** "The Collapse of the Indus-Script Thesis," *EJVS* 11 (2004) — the counter-position (IVS may be non-linguistic); cite to show you engage both sides.
5. **Shinde, V., et al.** "An Ancient Harappan Genome Lacks Ancestry from Steppe Pastoralists…," *Cell* 179 (2019) — aDNA anchor for the Dravidian/indigenous debate.
6. **Fuls, A.** positional/structural IVS analyses; **Wells, B.** on sign frequency/direction.

**(B) Computational epigraphy with CNNs/ML (the methods you extend)**
7. **Palaniappan, S., Adhikari, R.** "Deep Learning the Indus Script," arXiv:1702.00523 (2017) — two-CNN pipeline (region classifier + grapheme classifier); reports **92% accuracy** for detecting the *single* most frequent sign ("jar"); the direct precedent for your scalability argument.
8. **Mukhopadhyay, A., Chakraborty, S., Nasipuri, M., Das, N.** "Data Mining Ancient Script Image Data Using Convolutional Neural Networks," UNL CSE Conf. (2018) — cross-script CNN classification; strong methodological precedent + baseline to beat.
9. **Sproat, R.** on computational approaches to ancient scripts — segmentation & standardisation pitfalls.
10. **Indus Corpus / ICIT** (Indus Corpus of Inscriptions and Texts) — the standardised digital corpus you should train on *instead of* synthetic glyphs.
11. CNN/template pipelines for **Tamil-Brahmi & historical Tamil OCR** — for the deciphered side of your dual-read.

**(C) Methodology, statistics & caution (what makes results defensible)**
12. **Guo, C., Pleiss, G., Sun, Y., Weinberger, K.** "On Calibration of Modern Neural Networks," ICML 2017 — temperature scaling.
13. **Bendale, A., Boult, T.** "Towards Open Set Deep Networks," CVPR 2016 (OpenMax) — the open-set framework you need instead of closed-set softmax.
14. **Hendrycks, D., Gimpel, K.** "A Baseline for Detecting Misclassified and Out-of-Distribution Examples," ICLR 2017.
15. **Selvaraju, R., et al.** "Grad-CAM," ICCV 2017 — stroke-level explanations.
16. **"Signs Independent of Language and Meaning: A Probabilistic Law of Sign Convergence"** (arXiv, 2025) — cautionary: short sign inventories converge in shape by chance; cite to justify negative controls.
17. **Ganin, Y., Lempitsky, V.** "Domain-Adversarial Training of Neural Networks" (DANN), JMLR 2016 — principled fix for synthetic→archaeological domain shift.

### 6.4 Fast citation URLs
- Keeladi link news: `thenewsminute.com/tamil-nadu/major-discovery-tamil-nadu-s-keezhadi-possible-link-indus-valley-civilisation-109165`
- Deep Learning the Indus Script: `arxiv.org/abs/1702.00523`
- Sign-convergence caution: search arXiv for *"probabilistic law of sign convergence"*.
- Keeladi excavations (official): *tnarch.gov.in* (Dept. of Archaeology reports).

> ⚠️ The link you sent (`notebook.google.com/notebook/06372343-f2b3-4b92-928a-c9217cea9d66`) is a **private NotebookLM notebook** — it redirects to a Google sign-in page, so its contents are not publicly retrievable. To let me integrate it, export it to PDF/Google-Docs and share publicly, or paste the text here.

> 📌 Note on the "P-numbers" (P225/P307 etc.): these are **project-internal labels** for the 4 matched signs, not standard Mahadevan/Parpola sign numbers. In your paper, map them to the published sign IDs from Mahadevan's concordance so results are comparable to other work.

---

## 7. Roadmap to *genuinely* good accuracy

Order matters: **integrity → data → model → evaluation**. A better architecture on fake data/results just produces better-looking fakes.

### 7.1 Fix integrity first (do not skip — this is what makes "accuracy" meaningful)
1. **Delete all simulated metrics.** Remove the `np.random.*` blocks and hard-coded narratives from `run_pipeline.py` (§2.1), `process_model.py`, and `dashboard.py`. Reports must be *derived from model output only*.
   - `run_pipeline.py`: replace the L108–135 block with an import of the real trainer/evaluator; keep only the plotting.
   - `process_model.py`: replace L47–58 `"dummy but realistic metrics"` with a real load of `models/evaluation_results/*.json` produced by `evaluate.py`.
   - `dashboard.py`: replace the hard-coded `predictions_map` (L246–275) and `match_pct=85.0` default with a read of a real per-sherd JSON emitted by the evaluator.
2. **Fix the F1 formula** to the real harmonic mean: `F1 = 2*P*R/(P+R)`. Drop "(estimated)" precision/recall or compute them.
3. **Make every report traceable:** each number carries its source array/file and the run id.
4. **Seed everything** (`random`, `numpy`, `tf.keras.utils.set_random_seed`), record `pip freeze`, and hash the dataset.

### 7.2 Fix the data (this is where real accuracy actually comes from)
5. **Train on a real corpus, not synthetic glyphs.** Use the standardised **Indus Corpus / ICIT** (or digitised scans of Mahadevan's concordance) for the IVS side, and the actual Keeladi sherd photos for evaluation. Keep synthetic renderings only as an *augmentation* source, never as the only train set.
6. **Split by source *before* augmenting.** Replace the aug-then-split in `train.py` L288–293 with:
   - a `GroupKFold`/stratified split on **original glyph id** (or on *sherd id*), then augment only the training fold using `keras.layers` inside the model (as Keras augmentation layers) or a `tf.data` pipeline.
   - Reserve a **source-disjoint test set** touched exactly once.
7. **Add real negatives** to enable open-set evaluation: known non-sign marks (decoration, damage, blank sherds, unrelated graffiti), plus negatives from other scripts.
8. **Balance the classes** — either equalise counts, or train with class weights and report **macro-F1** and balanced accuracy.

### 7.3 Fix the modelling (so the task is the *right* task)
9. **Reframe recognition as an extreme-value / open-set problem**, not 49-way closed-set softmax: add an `unknown/other` class **with real negatives**, and score open-set-ness with **OpenMax** or an energy/entropy OOD score. Then "match" = (top-1 is an Indus sign) **and** (OOD score above a validated threshold).
10. **Consider metric learning for the *matching* part.** For "is this Keeladi sherd the same sign as X?", a **Siamese/triplet network** (embedding + cosine distance) generalises to unseen classes and gives a *verification* score with a threshold you can calibrate — much more defensible than a softmax label.
11. **Use the decomposition you already built.** Feed the 3×3 grid presence/symmetry features (`grid_decomposition.py`) into the model (as extra channels or a second input branch) instead of leaving it unused — this directly targets the "Decomposition" research gap.
12. **Domain adaptation instead of only augmentation.** Train with **DANN** (gradient-reversal) or MMD to align synthetic-IVS features with real-sherd features (`weight_transfer.py` already gestures at this with `evaluate_domain_gap`).
13. **Handle allographs explicitly.** Cluster the per-class embeddings and report intra-class (allograph) vs inter-class distances; use this to build the "transformation" narrative with numbers rather than hand-waving.
14. **Improve the architecture modestly**: the current 6-conv VGG-ish net is fine, but add **global average pooling** instead of a large `Flatten` (fewer params, less overfit), and consider a small **ResNet/EfficientNet-B0** transfer backbone for the real-image stage. Input 64×64 loses stroke detail — use **128×128**.

### 7.4 Fix the evaluation (to report defensible numbers)
15. **Cross-validate** (5-fold, group-aware) and report **mean ± 95% CI** for accuracy, macro-F1, top-3.
16. **Calibrate** with temperature scaling; report a **reliability diagram + ECE** so a "0.8 confidence" means ~80% correct.
17. **Baselines & ablations**: (a) HOG/SIFT/Zernike + SVM; (b) template matching (you already have a Brahmi template matcher in `decoding.py`); (c) a human-expert pair-verification task with **Cohen's κ** agreement. Ablate: with/without autocrop, with/without grid features, with/without domain adaptation.
18. **Significance testing** for the headline claim: permutation/bootstrap test of "matched pairs are more similar than random pairs", and report **effect size** (e.g. AUC, Cohen's d), not just a percentage.
19. **Report a confusion matrix + per-class metrics** (`evaluate.py` already builds these — make them the *primary* result, not the fabricated summary).

### 7.5 The accuracy ladder (expectations, honestly)
| Data / protocol | Realistic outcome |
|---|---|
| Current (synthetic train + leaked val) | 90–97% val — **meaningless** |
| Real IVS glyphs, group-split, closed-set | ~85–95% top-1 on in-distribution signs |
| + open-set + negatives | report **AUROC** (aim >0.9) and *rejection rate*; top-1 only on accepted inputs |
| + domain adaptation to real sherds | the number you can actually defend for Keeladi matching |

**Rule of thumb:** if the reported accuracy is above ~97% on this problem with n≈900 images, suspect leakage/synthetic data before celebrating.

---

## 8. Prioritised action checklist (mapped to code)

| # | Priority | Action | Where |
|---|---|---|---|
| 1 | 🔴 P0 | Remove all simulated metrics | `run_pipeline.py` L108–135, L311–405; `process_model.py` L46–58, L141, L166–167; `dashboard.py` L158–159, L246–275 |
| 2 | 🔴 P0 | Fix augment-before-split leakage | `src/train.py` L288–293 |
| 3 | 🔴 P0 | Fix F1 (harmonic mean) + drop "estimated" P/R | `run_pipeline.py` L326–328 |
| 4 | 🔴 P0 | Replace hard-coded "decodings" with model output or rename them as *reference readings* | `cnn_annotation_generator.py` L17–67; `cnn_potsherd_annotator.py` L151+ |
| 5 | 🟠 P1 | Train on real corpus (ICIT/Mahadevan digitisation), not just synthetic glyphs | `data/processed/train/*` |
| 6 | 🟠 P1 | Add `zz_/unknown` class **with real negatives**; wire up `REJECTION_PREFIX` in `evaluate.py` | `evaluate.py` L39, `src/train.py` |
| 7 | 🟠 P1 | Group-aware 5-fold CV + report mean±CI, macro-F1, top-3 | `src/train.py`, `src/evaluate.py` |
| 8 | 🟠 P1 | Calibrate confidences (temperature scaling) + ECE/reliability diagram | new `src/calibration.py`; `evaluate.py` |
| 9 | 🟡 P2 | Open-set scoring (OpenMax / energy) instead of pure softmax | `src/models/indus_classifier_cnn.py`, `evaluate.py` |
| 10 | 🟡 P2 | Siamese/triplet verification head for matching | `src/models/` (new `siamese_matcher.py`) |
| 11 | 🟡 P2 | Feature the 3×3 grid decomposition into the model | `src/preprocessing/grid_decomposition.py` + model |
| 12 | 🟡 P2 | Domain-adversarial adaptation (DANN) synthetic→real | `src/models/weight_transfer.py` |
| 13 | 🟢 P3 | Baselines (HOG/Zernike+SVM, template match) + expert κ | new `src/baselines.py` |
| 14 | 🟢 P3 | Significance tests + effect sizes | new `src/stats.py` |
| 15 | 🟢 P3 | Seed/version/data-hash manifest for reproducibility | `src/__init__.py` / `run_pipeline.py` |
| 16 | 🟢 P3 | Map project P-numbers → Mahadevan sign IDs | `data/lexicon.json`, docs |

---

## 9. Bottom line

- The **architecture, preprocessing and decomposition code are reasonable**, and the research *idea* (quantifying the Indus→graffiti→Brahmi link) is publication-worthy.
- What makes the current results **invalid** is not the CNN — it is (1) **fabricated metrics**, (2) **augmentation leakage**, (3) **synthetic-only data**, (4) **closed-set framing with no negatives**, and (5) **an inference leap from shape similarity to script descent**.
- **Good accuracy** here does not come from a fancier CNN. It comes from *real data + a leak-free protocol + open-set rejection + calibration + statistical reporting*. Fix those and 85–95% (in-distribution, macro-F1) plus a defensible open-set AUROC is a realistic, publishable target.
- The **connecting link** to cite is the Keeladi Excavation Report (2019) "missing link" hypothesis + the Dravidian hypothesis (Mahadevan/Parpola), tested computationally in the spirit of **Palaniappan & Adhikari (arXiv:1702.00523)** and **Mukhopadhyay et al. (2018)**, with the caution of the **sign-convergence** literature baked in as negative controls.

> Reminder: the NotebookLM link is private. Share the text and I'll fold it into the problem statement and reference list.









---

## 10. Measured results after the fixes (what the honest pipeline produced)

After removing the fabricated metrics and adding `src/audit_validity.py`, the
pipeline was run end-to-end (`python run_pipeline.py`). These are the **real**
numbers it wrote to `models/evaluation_results/validity_audit_report.txt`:

**Leakage audit**
| protocol | n_train | n_val | % val with ≥0.99 twin in train | mean max similarity |
|---|---|---|---|---|
| random split (old behaviour) | 833 | 147 | **10.88%** | **0.9512** |
| source-disjoint split | 800 | 180 | 2.78% | 0.9192 |

→ Verdict: **CONTAMINATED** — the old validation accuracy mostly measured
memorisation of augmented clones.

**Open-set audit (false-acceptance)**
| group | n | mean top-1 | % ≥0.5 | % ≥0.8 |
|---|---|---|---|---|
| negative_blank | 5 | **0.9989** | 100 | 100 |
| negative_saturated | 5 | 0.9911 | 100 | 100 |
| negative_noise | 20 | 0.5470 | 75 | 0 |
| negative_tamil_brahmi_letters | 37 | 0.3967 | 21.6 | 2.7 |
| general_keeladi_graffiti | 12 | 0.5536 | 50 | 16.7 |
| positive_indus_train | 100 | 0.5602 | 52 | 21 |

→ Verdict: negatives are scored **as confidently as positives** (blank = 99.9%).
The closed-set model cannot say "not a sign".

**Verification audit (the 4 hand-picked pairs)**
| sherd | expected sign | P(expected) | model top-1 | correct |
|---|---|---|---|---|
| match_Indus_225 | sign_25_P225_Cross | 0.0337 | sign_32_P296 (0.42) | ✗ |
| match_Indus_307 | sign_41_P307 | 0.0011 | sign_25_P225_Cross (0.57) | ✗ |
| match_Indus_318 | sign_42_P318 | 0.0006 | sign_04_P76_Fish (0.73) | ✗ |
| match_Indus_365 | sign_43_P365 | 0.0000 | sign_29_P278 (0.58) | ✗ |

→ 0/4 top-1, 0/4 in top-3; mean P(expected)=0.0089 vs null mean 0.0202,
**permutation p = 0.303**. The hand-picked matches are **not reproduced** by the
model — no better than chance.

**Real Keeladi evaluation** (`src/evaluate.py`): 37 images, mean confidence
0.4500, match rate @0.5 = 29.73% (11/37), known-pair top-1 0/4.

### 10.1 The discovery in one sentence
> On this dataset the CNN's apparent success was entirely an artefact: once the
> fabricated metrics are removed and the split is made honest, the model neither
> reproduces the motivating Indus↔Keeladi matches (p≈0.30) nor rejects non-signs
> (blank images "recognised" at 99.9% confidence) — which reframes the problem
> from "prove the link" to "build a dataset and protocol that could ever test it".

### 10.2 Files added/changed by this work
| File | Change |
|---|---|
| `src/audit_validity.py` | **new** — leakage / open-set / verification audit |
| `src/train.py` | split-before-augment (leakage fix); `split_protocol` option |
| `run_pipeline.py` | rewritten: no `np.random`, model-derived only |
| `process_model.py` | rewritten: reads real results, invents nothing |
| `dashboard.py` | hard-coded metrics/predictions replaced with real JSON reads |
| `cnn_annotation_generator.py`, `cnn_potsherd_annotator.py` | readings marked as authored reference, not CNN output |
| `README.md` | fabricated "expected results" replaced with measured numbers |
| `models/evaluation_results/validity_audit_report.txt` (+ `.json`) | **new** real audit output |
| `models/evaluation_results/keeladi_predictions.json` | **new** real per-sherd top-3 |


---

## 11. Ground truth verified from the source PDFs (`docs/*.pdf`)

Two source documents were added to `docs/`. Both extracted cleanly
(`pypdf`; see the generated `.extracted.txt` files) and **fully corroborate** the
project's premises — while exposing one concrete, fixable modeling error.

### 11.1 What the sources confirm ✓
| Project claim | Source (verified) |
|---|---|
| 40 primary core signs | *THE INDUS SCRIPT — Recognition as an Alphabet*: "a core set of only **40** fundamental signs"; conclusion table: *Primary Core Signs 40* |
| 404 signs examined | same paper: *examined 404* (Parpola 391 + Wells + Mahadevan) |
| Grid decomposition technique | same paper: "novel **grid-based decomposition technique**" (Fig. 01) |
| 3 permanent modifiers | same paper: "**Three permanent Modifiers**" + conclusion "**3 principal modifiers**" |
| 8 diacritical marks (on disk) | same paper: conclusion "**8 diacritics**" |
| 1,001 graffiti sherds | *keeladi_indus.pdf*: "the recovery of **1,001 graffiti sherds** from Keeladi" |
| 56 Tamil-Brahmi sherds | same: "At Keeladi, **56** Tamil-Brahmi inscribed potsherds" |
| Names *kuviran atan* / *atan* | same: personal names "**kuvira-atan**" and "**ātan**" → the project's corpus readings are correct |
| The 4 matched signs 225/307/365/318 | Annexure, p.63: "**similarities between graffiti symbols of Keeladi and signs of Indus Civilization** — indus sign-225, sign-307, sign-365, sign-318, sign-318" |
| `sign_01_P13` … `sign_40_P120` | Fig. 65 table column **"P-2010"** matches the class names exactly |
| The continuity hypothesis itself | Keeladi report: graffiti "evolved or transformed from Indus script … precursor for Brahmi" |

Note the annexure's own word is **"similarities"**, not "identity" or "proof" —
which matches the disciplined framing in §6 of this document.

### 11.2 🔴 A concrete, fixable error: 45 classes should be 40
Figure 65 lists **40 serial signs**. Several rows give alternative codes joined by
"or" (i.e. *the same sign*, an allograph or alternate reading):

| Fig.65 serial | P-2010 alternatives | Class folders created |
|---|---|---|
| 17 | 156 or 165 | `sign_17_P156_P165` → **1** ✓ merged |
| 18 | 181 or 187 | `sign_18_P181` + `sign_18_P187` → **2** ✗ split |
| 21 | 200 or 209 | `sign_21_P200` + `sign_21_P209` → **2** ✗ split |
| 28 | 272 or 371 | `sign_28_P272` + `sign_28_P371` → **2** ✗ split |
| 30 | 282 or 285 or 287 | `sign_30_P282`+`P285`+`P287` → **3** ✗ split |

36 single serials + (1+2+2+3) = **45** — exactly the 45 `primary_core_signs`
folders (49 classes total with the 4 `indus_matched`). The 5-class inflation comes
**only** from serials 18, 21, 28, 30 being split.

This is precisely Research Gap 3 (allograph vs. distinct sign) showing up as a
**training bug**: `sign_30_P282/P285/P287` are *the same sign*, so the model is
forced to separate near-identical images into mutually-confusing classes.

**Fix (concrete accuracy win):** merge those 4 groups back → **45 → 40 classes**,
so the label space matches the source's alphabet. This should cut confusion among
the 8 near-duplicate images and make class names comparable to other work.

> ⚠️ (`sign_12_P130` vs `sign_13_P130_variant`, and `sign_35_P341_Oval` vs
> `sign_36_P341_Leaf`, are **genuinely different serials** that merely share a
> P-number — do NOT merge those.)

### 11.3 ⚠️ To verify before trusting `EXPECTED_MATCH_MAP`
The annexure numbers (225, 307, 318, 365) are ambiguous because **both** the
P-2010 and Mahadevan (M-1977) columns use values in that range:
- **P-2010 = 225** → Fig.65 serial 25, *but* **M-1977 = 225** → serial 24 (P=219).
- **M-1977 = 307** → serial 18 (= P-2010 181/187), while no P-2010 entry equals 307.

So `EXPECTED_MATCH_MAP` may be pointing at the wrong sign. Confirm which numbering
the Keeladi annexure uses (read the figure captions on PDF p.63) before treating
`0/4 top-1` as a model failure rather than a mislabelled expectation.

### 11.4 Also worth noting
- The Indus paper's conclusion asserts a **Prakrit / Indo-European** reading, whereas
  `data/lexicon.json` adopts **Parpola/Mahadevan-style (Dravidian)** readings. Pick one
  and state it; do not mix them in a report.
- Fig. 65 includes **NFM Unicode PUA** codepoints (E06-D, E10-A, …). If the NFM font is
  available, the 40 signs can be **rendered programmatically** → a real, expandable
  training set instead of the current 1 drawing + 19 synthetic clones.

---

## 12. RESOLVED: the Keeladi annexure uses MAHADEVAN numbering (visual verification)

§11.3 flagged an ambiguity. It is now **settled** by rendering the actual annexure
plate and the Figure-65 table and comparing glyph shapes.

### 12.1 What the annexure actually shows
`keeladi_indus.pdf`, p.63, plate headed *"SIMILARITIES BETWEEN GRAFFITI SYMBOLS OF
KEELADI AND SIGN OF INDUS CIVILIZATION"*, columns *Keeladi graffiti | INDUS sign*:

| # | Keeladi graffiti | INDUS sign glyph | label |
|---|---|---|---|
| 1 | sherd, small angular mark | ▷ with X across | INDUS SIGN-225 |
| 2 | sherd, arrow through D | D with diagonal slash | INDUS SIGN-307 |
| 3 | sherd, fan/arrow | V with centre stroke | INDUS SIGN-365 |
| 4 | sherd, zigzag | arch + crossbars + chevron | INDUS SIGN-318 |
| 5 | sherd, radiating lines | double-forked U (two plants in a cup) | INDUS SIGN-318 |

⚠️ **The report labels rows 4 AND 5 with the same number "318", yet they are
---

## 15. Figure 65 digitisation + metric learning (implemented)

This section records the two items that §14.5 said were the "next real gain".

### 15.1 What was digitised

`src/digitise_fig65.py` renders **Figure 65 (pp. 35–36)** of
`docs/THE INDUS SCRIPT Recognition as an Alphabet.pdf` at 6× and crops the
"Indus Sign" cell of each row.

- **39 real sign glyphs** recovered (serials 1–39; serial 40 is a photograph of a
  seal, not a drawing, so it is correctly skipped).
- Cell boxes are derived from the PDF's **vector table rules** (`get_drawings`),
  so the crop sits strictly inside the cell and no table border is included.
- Each glyph is labelled by serial, NFM PUA code, **M-1977 (Mahadevan)**,
  W-2015 and **P-2010** (see `FIG65_ROWS`).
- Output: `data/processed/train/fig65_real/*.png` + `manifest.json`.
- QA artifact: **`docs/figures/fig65_digitised_contact_sheet.png`** — a labelled
  grid of all 39 glyphs. All 39 match the published table.

This replaces the previous training data, which was *one drawing per sign plus
synthetic clones*.

### 15.2 Metric learning

`src/siamese_embed.py` trains a small CNN with a **64-D L2-normalised embedding**
using a **batch-hard triplet loss** plus a classification head, on the real
Fig. 65 glyphs with random affine/noise perturbations. Batches are
class-balanced (6 instances × 12 classes) so every anchor has a positive and
hard negatives.

Retrieval of held-out perturbed queries against a clean gallery
(`models/evaluation_results/siamese_report.json`):

| Metric | Result | Chance |
|---|---|---|
| Top-1 | **0.859** | 0.026 |
| Top-5 | **0.974** | — |
| Mean rank | **1.41** | 20.0 |

### 15.3 The stroke-sign problem is now measured, not assumed

Per-class top-1 splits the 39 signs cleanly:

| Group | Mean top-1 |
|---|---|
| Non-stroke signs | **1.000** |
| Stroke signs (serials 9–16, 30) | **0.389** |

Five signs score **0.000** top-1 — `row09 (P-127)`, `row10 (P-128)`,
`row14 (P-133)`, `row15 (P-145)`, `row16 (P-147)`. These are all plain vertical
strokes that differ only in stroke count and position, so under any
shape-similarity metric they are **not separable**. This is direct evidence
for the §14.4 caveat, and confirms that stroke signs must be excluded from
match claims rather than merely down-weighted.

`STROKE_SERIALS` in `src/digitise_fig65.py` and the `is_stroke_sign` flag in
the manifest carry this classification forward.

### 15.4 Honest limits of these numbers

- The held-out queries are **perturbations of the same source glyphs** used for
  training. This measures **robustness to geometric perturbation only**. It is
  an **upper bound**, not evidence of performance on real, independently
  engraved signs.
- The embedding is trained on **39 classes**, not the full 40-sign inventory,
  and is **not yet wired into `src/sign_matcher.py`** for the four-pair
  verification. That integration is the remaining step.
- Nothing here bears on the historical question of Indus–Keeladi script
  continuity. These are shape statistics only.
different signs.** That is an error in the source plate, not in our code — which
is exactly why two classes (`sign_42_P318`, `sign_42_P318b`) exist.

### 12.2 Proof of the numbering system
| Annexure label | Fig.65 serial | That serial's M-1977 | That serial's P-2010 |
|---|---|---|---|
| 225 | **24** (glyph ▷+X matches) | **225** ✓ | 219 |
| 307 | **18** (glyph D+slash matches) | **304 or 307†** ✓ | 181 or 187 |

No P-2010 entry equals 307 or 365; both appear in the **M-1977** column.
**⇒ the annexure is numbered by Mahadevan (M-1977), not by P-2010.**

Consequences that are now **fixed in code**:
- `match_Indus_225` expected `sign_25_P225_Cross` → **wrong sign**; correct class is `sign_24_P219`.
- `match_Indus_307` expected `sign_41_P307` → **wrong sign**; correct class is serial 18 (`sign_18_P181_P187`).
- 318 / 365 are outside the 40 core signs, so their annexure-only classes are kept (self-consistent pairing).

### 12.3 Changes implemented (non-destructive — nothing on disk was moved)

| Change | Where | Effect |
|---|---|---|
| Allograph merge (Fig.65 "or" rows 18, 21, 28, 30) | `src/train.py` `ALLOGRAPH_GROUPS` / `canonical_class()` | 45 → **40** core classes; removes 5 near-duplicate classes |
| Annexure remap (M-1977 → correct core class) | `src/train.py` `ANNEXURE_REMAP` | fixes label noise where two different glyphs shared `sign_25_P225_Cross` |
| Expected-match map corrected + made multi-valued | `src/evaluate.py`, `src/audit_validity.py` | a match counts if **any** acceptable allograph is predicted |
| Training schedule configurable | `src/train.py` `INDUS_AUG` / `INDUS_EPOCHS` / `INDUS_BATCH` | fast CPU runs and full runs, both reproducible |

**Result: 43 classes** = 40 core signs + 3 annexure-only (318, 318b, 365).
The old 49-class model is preserved as `models/legacy49_indus_classifier.keras`.

### 12.4 Known data gap
`sign_40_P120_SemiSigns` (Fig.65 serial 40) contains **0 images** — the class is
declared but never trained. Add real scans, or drop the class, before publishing.
---

## 13. After the fixes: retrained model, measured results (BEFORE → AFTER)

Retrained with the corrected label space (43 classes) and the leak-controlled
split (`INDUS_AUG=6 INDUS_EPOCHS=25 INDUS_BATCH=32`, CPU run).
Legacy 49-class weights preserved as `models/legacy49_indus_classifier.keras`.

**Validation (honest, held-out originals never augmented)**
| | old (leaky) | new (leak-controlled) |
|---|---|---|
| Final val accuracy | ~0.90+ (inflated) | **0.7279** |
| Best val accuracy | — | **0.7755** (epoch 20) |

**Open-set false acceptance — the biggest win**
| group | old mean top-1 | new mean top-1 | new % ≥0.5 |
|---|---|---|---|
| negative_blank | 0.9989 (100% FA) | **0.1017** | **0.0%** |
| negative_saturated | 0.9911 (100% FA) | **0.1943** | **0.0%** |
| negative_noise | 0.5470 | **0.1089** | **0.0%** |
| negative_tamil_brahmi_letters | 0.3967 | 0.4256 | 35.1% |
| **positive_indus_train** | 0.5602 | **0.7945** | 79.0% |

> Old model: negatives scored **higher** than positives (blank 0.999 vs 0.560) —
> it could not tell a sign from a blank page.
> New model: positives **0.79** vs negatives **0.10–0.19** — clean separation.
> This improvement came from fixing the *labels* (allograph merge + annexure
> remap), not from a bigger network.

**Leakage audit** (measures what the OLD random protocol would give):
10.20% of val images have a ≥0.99 twin in train (mean max sim 0.9469);
source-disjoint split → 5.00%.

**Verification of the 4 hand-picked pairs (corrected expectations)**
| sherd | expected (corrected) | P(expected) | model top-1 | correct |
|---|---|---|---|---|
| 225 | `sign_24_P219` | 0.0187 | sign_32_P296 (0.47) | ✗ |
| 307 | `sign_18_P181_P187` | 0.0358 | sign_21_P200_P209 (0.44) | ✗ |
| 318 | `sign_42_P318` / `sign_42_P318b` | 0.0107 | sign_23_P217 (0.50) | ✗ |
| 365 | `sign_43_P365` | 0.0008 | sign_32_P296 (0.46) | ✗ |

0/4 top-1, mean P(expected)=0.0165 vs null 0.0224, **p = 0.264**.

**Keeladi run**: 37 images, mean confidence 0.4787, match rate **35.14%** (13/37).

### 13.1 What still blocks a positive result
1. **The model now works, but the sherd drawings do not resemble the sign
   glyphs.** On genuine Indus training images it is confident (0.79) and
   confident these Keeladi marks are *not* the paired signs. With p = 0.264
   there is no evidence of the 4 published correspondences **in this data**.
2. **`sign_40_P120_SemiSigns` has 0 images** — serial 40 is untrained.
3. **Training data is still synthetic**: 1 drawing per source + augmented
   clones. The Keeladi sherds are real photos. That domain gap is the main
   remaining source of the failure.
4. **n = 4.** Four hand-picked pairs cannot establish or refute a civilisational
   link; only the full 1,001-sherd corpus with verified labels could.

### 13.2 Recommended next step (highest value first)
Get real, labelled Indus scans — digitising **Fig. 65** (40 signs) and the
**Fig. 59** allograph plate from `docs/THE INDUS SCRIPT …pdf`, using the NFM
Unicode PUA codepoints listed in Fig. 65 to render the glyphs. That single step
replaces 19 synthetic clones per sign with genuine allographic variety and
should let the leak-controlled accuracy rise well above the current 0.73.
---

## 14. Why the 4 Keeladi samples scored only 40-50% — and the fix

### 14.1 Root cause: the wrong instrument, plus a preprocessing bug
`src/evaluate.py` answered a **closed-set classification** question ("which of the
43 classes is this image?") and reported its **softmax** as a match score. Two
consequences:

1. A softmax over 43 classes is *not* a similarity measure. Even a perfect
   visual match rarely yields P > 0.5, so genuine pairs landed at 0.40-0.50.
2. A Keeladi "match" file is a **potsherd photo**: it contains the pot outline
   and rim line *as well as* the incised glyph. The shared `ImageNormalizer`
   auto-crops to the **largest contour**, which is usually the *pot outline*, not
   the glyph. The network was therefore classifying the wrong object.

The classical path (`generate_known_pair_comparison`) already segmented the sherd
into glyphs first and scored 59-73% on the same pairs — i.e. **the CNN was
underperforming a trivial baseline on its own headline task.**

### 14.2 New module: `src/sign_matcher.py` (open-set verification / retrieval)
For each sherd it now:
1. **segments** the sherd into individual glyphs (`InscriptionDecoder`, outline
   excluded), so the pot rim is never classified;
2. extracts the **128-d penultimate embedding** from the trained CNN;
3. computes **cosine similarity** to a per-class prototype;
4. computes **dilation-tolerant bidirectional stroke coverage** against each
   class's glyphs (robust to thin scratches vs. thick catalogue strokes);
5. combines the two, and reports the expected sign's **rank** + a permutation
   test.

Each class keeps its best-matching glyph, so stray strokes cannot dilute a score.
Wired into `run_pipeline.py` as step **[4/6]**.

### 14.3 Measured result (same 4 sherds, same images, nothing hand-tuned)
| sherd | old softmax "match" | **new verification score** | rank /43 | in top-5 |
|---|---|---|---|---|
| match_Indus_225 | ~0.47 | **63.96%** | **4** | ✅ |
| match_Indus_307 | ~0.44 | **64.73%** | 23 | — |
| match_Indus_318 | ~0.50 | **71.49%** | 9 | — |
| match_Indus_365 | ~0.46 | **63.11%** | 16 | — |

- mean expected score **65.82%** vs random-class mean **57.36%**
- **1/4 in top-5** (was 0/4) · permutation **p = 0.1025** (was 0.264)
- Now in the same band as the classical baseline (59-73%).

**This is a real methodological improvement, not a tuned number.** Scores went up
because the matcher stopped measuring the wrong thing.

### 14.4 Honest caveats
- **p = 0.1025 is still not significant** at the conventional 0.05 level, and
  n = 4. The evidence for the published correspondences is *suggestive, not
  established*.
- **Stroke-coverage degeneracy:** for simple stroke signs (e.g. `sign_11_P129`,
  two vertical strokes) coverage reaches 100% against almost any thin stroke,
  which is why `match_Indus_365` scores ~100 on a stroke sign. Stroke signs
  should be excluded from shape-matching claims (this is the same
  sign-convergence problem noted in §6.3 item 16).
- Cross-contamination between annexure pairs is visible (the 307 sherd's top-1 is
  `sign_42_P318`), so the four examples are not independent.
- A higher score means **the shapes agree**. It is still not evidence of script
  descent.

### 14.5 Next real gain
Train the embedding with a **metric (triplet/siamese) loss on real allographs**,
and drop stroke signs from the comparison set. Both need the Fig. 59 digitisation
described in §13.2.
