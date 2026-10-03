# SEM QC pipeline evaluation — shared spec (lead-authored, authoritative)

All workstreams MUST follow this file. If something here is wrong or blocks you, stop and report — do not silently deviate.
New code lives under `sem/qc/` (to avoid colliding with the unmerged `devin/1791031501-sem-pipeline` branch, called **pipeline_v0** below, which owns `sem/*.py`).
Python 3.10+, CPU only. No Modal. Tests: `pytest -q` (CPU, < 2 min, synthetic fixtures; never require the real dataset).

## 0. Data facts (verified)
- Source: public Drive folder `12UnB4HYDElXzoR4I0mG7NZ0buSr4QXF6` → `gdown --folder https://drive.google.com/drive/folders/12UnB4HYDElXzoR4I0mG7NZ0buSr4QXF6 -O /home/ubuntu/data/sem`.
  Layout: `/home/ubuntu/data/sem/Batch_{1,2,3}/img_<stem>_<VIEW>.tif`. 93 files, 31 stems (Batch_1: 7, Batch_2: 7, Batch_3: 17).
- Data root is configurable (`SEM_DATA_ROOT` env var, default `/home/ubuntu/data/sem`). Raw files are read-only. Derived outputs go to `SEM_WORK_ROOT` (default `/home/ubuntu/work/qc`), never into git.
- Views: BSE, Inlens always present; ETD or SE (detector_set "ETD" or "SE"). **Common views used everywhere: BSE (primary) + Inlens.** ETD/SE are never model inputs.
- Images are 8-bit gray stored as RGB (use channel 0), 7000 (or 6996) wide × 1612–2272 tall. Views of one stem are co-registered.
- Pixel size 25 nm/px from TIFF XResolution (inch units). Parse it per file; assert 24.9–25.1 nm, else raise.
- 1–4 px coloured columns at left/right edges of some images (stitching). `valid_mask` = NOT(any pixel where channels differ, dilated by 2 px) AND NOT(8 px border on every side).
- Material: graphite-flake anode with bright high-Z particles; non-infiltrated cross-section → sub-surface material visible inside pores (darkness ≠ pore).
- All stems are from the SAME physical sample. Batches therefore most likely differ by acquisition session, not material.

## 1. Taxonomy (`sem/qc/schema.py`)
Pixel classes (uint8 label maps):
| id | name | notes |
|---|---|---|
| 0 | matrix_other | binder/carbon-black/mottled filler, anything solid not below |
| 1 | graphite_particle | dark-grey flakes |
| 2 | bright_particle | bright high-Z particles |
| 3 | pore | open in-plane void (true hole at the cut plane) |
| 4 | subsurface_uncertain | inside a pore but showing out-of-plane material, or undecidable dark region |
| 5 | crack_intraparticle | fracture inside one particle |
| 6 | interparticle_gap | thin separation along particle/particle or particle/matrix boundary |
| 7 | artifact | acquisition/prep artifact pixels (subtype on instance) |
| 255 | ignore | unlabeled / invalid / outside valid_mask |
Instance-level only (polygons): `agglomerate` (tight cluster of ≥3 bright particles, polygon = cluster hull) and artifact instances with `subtype ∈ {curtaining, scan_streak, charging, redeposition, edge_column, other}`.
Mapping pipeline_v0 → qc: crack_intra→5, crack_inter→6, void→3, agglomerate→agglomerate instance, curtaining/edge_bloom→artifact(7, subtype curtaining/charging), other_anomaly→unmapped (reported separately), background→none.

## 2. Interfaces
### 2.1 Loading (`sem/qc/io.py`)
`list_stems(root) -> list[StemRecord]` (stem, batch, detector_set, views{view: path}, height, width, pixel_size_nm).
`load_stem(rec, views=("BSE","Inlens")) -> dict[str, np.ndarray uint8 HxW]`; `valid_mask(rec) -> bool HxW`.
### 2.2 Predictions (every method)
Protocol `Method`: `name: str`; `predict(views: dict[str, np.ndarray], valid: np.ndarray) -> Prediction`.
`Prediction`: `semantic` uint8 HxW (ids above, 255 outside valid), `instances: list[Instance]`, `uncertainty` float32 HxW in [0,1] (or None).
`Instance`: `class_name` (one of pixel class names or "agglomerate"), `subtype` (artifacts) , `bbox` [x0,y0,x1,y1] full-res, `polygon` list[[x,y]] full-res, `score` float in [0,1], `source` str.
On disk: `$SEM_WORK_ROOT/preds/<method>/<stem>_semantic.png`, `<stem>_uncertainty.png` (uint8 = round(255·u)), `<stem>_instances.json`.
Methods: `classical_v1` (foundation), `unet_pseudo_v1`, `pipeline_v0_proposals`, `vlm_claude_crops` (candidate-class only). Model/VLM outputs are NEVER ground truth.
### 2.3 Review items / labels (`sem/qc/schema.py`, JSON)
```
ReviewItem {item_id (sha1(stem|kind|x0,y0,w,h|source)[:12]), stem, batch, split, kind: "exhaustive_tile"|"candidate",
  tile {x0,y0,w,h} full-res, sampling {stratum: str, method: "random"|"uncertainty", weight: float},
  proposal {source, class_name, subtype|null, polygon|null, semantic_png|null, score, uncertainty} ,
  vlm_suggestion {model_id, class_name, confidence, rationale}|null,     # display only, never GT
  human {status: null|"accepted"|"rejected"|"relabeled"|"redrawn"|"uncertain", class_name|null, subtype|null,
         polygons: [{class_name, subtype, points}], semantic_png|null, notes, reviewer_id, timestamp}|null}
is_ground_truth(item) == human.status in {"accepted","relabeled","redrawn"}   # the ONLY GT predicate; used by eval
```
Store: `$SEM_WORK_ROOT/review/items.jsonl` (proposals, immutable) + `$SEM_WORK_ROOT/review/decisions.jsonl` (append-only human decisions, latest per item_id wins) + `review/masks/<item_id>.png`.
Exhaustive tiles: 512×512 at full res, crop from BSE (Inlens shown side-by-side), human edits a pre-filled label map (from classical_v1) with brush/polygon fill per class; status "accepted" = pre-fill correct as is, "redrawn" = edited.
### 2.4 QC quantities (`sem/qc/kpi.py`) — single implementation used by eval AND drift
Inputs: semantic map, instances, valid mask (and ignore), pixel_size_nm. Valid area A = pixels not 255 and not artifact(7).
- `void_fraction` = (#pore + #interparticle_gap) / A. `void_fraction_incl_uncertain` = (#3+#4+#6)/A (sensitivity variant).
- `bright_particle_ecd_um`: connected components (8-conn) of class 2, area ≥ 16 px, excluding components touching the tile/image border or 255; ECD = 2·sqrt(area/π)·px_um. Report count, median, p10, p90. Graphite: area fraction only (touching flakes → no instance sizes; documented limitation).
- `crack_density_um_per_mm2` = skeleton length of class 5 (px·px_um, skimage skeletonize) / (A·px_um²·1e-6). Same for class 6 → `gap_density_um_per_mm2`.
- `agglomerate_per_mm2` = #agglomerate instances (centroid in valid) / (A in mm²); `agglomerate_area_frac` = polygon area ∩ valid / A.
- `area_frac_<class>` for classes 0–7.
All 2D section measurements; no stereology.

## 3. Split (`sem/qc/split.py`) — lead-defined algorithm
Group = stem, except stems the similarity check (`scripts/stem_similarity.py`) flags as overlapping/adjacent, which the lead merges via `configs/groups_override.yaml`.
Strata = batch × detector_set. Within a stratum order groups by sha256(f"{seed}:{group_id}") hex ascending; n = #groups;
n_test = round(0.2n), n_val = round(0.2n); if n ≥ 3 ensure n_test ≥ 1 and n_val ≥ 1; first n_test → test, next n_val → val, rest → train. seed = 20261003.
Manifest `data/splits/manifest.csv` (committed): stem, group_id, batch, detector_set, split, views, height, width, pixel_size_nm, n_nongray_px, files_sha256 (";"-joined). Plus `data/splits/manifest_summary.md`.
Rules: GT review items only from val+test stems; headline metrics on test; val only for threshold/hyper-parameter choices; training only on train stems. Split is FROZEN after human approval (checkpoint 1); `split.py` must assert the committed manifest hash on load.

## 4. Evaluation (`sem/qc/eval/`)
Only `is_ground_truth` items. Headline metrics = random-stratum items on test stems; uncertainty-stratum and val reported separately.
- Pixel (exhaustive tiles): per class TP/FP/FN pooled over tiles; precision, recall, F1, IoU; ignore 255; GT class 4 pixels excluded from pore(3) scoring (neither TP nor FP). Row-normalized confusion matrix 8×8 (GT rows, pred cols) + raw counts.
- Object: instances via connected components (8-conn) per class for 2,3,5,6; agglomerates from polygons. Hungarian matching on IoU; match thresholds: 0.5 for classes 2,3; 0.3 for 5,6 and agglomerate. Object P/R/F1.
- Candidate items: per method/source/class precision = accepted / (accepted + rejected + relabeled-to-different-class); "uncertain" excluded and counted; Wilson 95% CI. Recall NOT computed from candidates.
- QC quantity error per tile and per stem (§2.4 on GT vs pred): bias, MAE, relative error, Spearman ρ across tiles.
- Uncertainty: stem-level bootstrap (resample stems with replacement, 2000 reps, seed 0) 95% percentile CIs for every metric.
- FP/FN galleries: per class top-10 FP and top-10 FN regions by area, PNG panels (BSE crop | GT overlay | pred overlay).
Outputs: `$SEM_WORK_ROOT/eval/<method>/metrics.json`, `tables/*.csv`, `galleries/*.png`, `report.md`.

## 5. Batch change detection (`sem/qc/drift/`)
Unit = stem. Reference = Batch_1; incoming = Batch_2, Batch_3 (each tested separately). Same sample → expect acquisition-driven differences.
Tiles: non-overlapping 1024×1024 within valid_mask (≥90% valid), features per tile, stem vector = median over its tiles.
Feature vector (interpretable; names fixed):
- material (from classical_v1 via §2.4 kpi.py, BSE): void_fraction, void_fraction_incl_uncertain, area_frac_bright_particle, area_frac_graphite_particle, bright_particle_ecd_median_um, bright_particle_ecd_p90_um, crack_density_um_per_mm2, gap_density_um_per_mm2, agglomerate_per_mm2.
- acquisition covariates: bse_mean, bse_std, inlens_mean, inlens_std, bse_focus (var(Laplacian)/var(img)), bse_noise (1.4826·MAD of img − gaussian(img, σ=1)), curtaining_score (pipeline_v0 fft def), scan_streak_score (std of row-means of high-passed img / img std), edge_column_px.
Methods:
1. Per-KPI: robust z = (median_inc − median_ref)/(1.4826·MAD_ref) (MAD=0 → std; both 0 → "degenerate"); Hedges' g; difference of medians with stem-bootstrap 95% CI (2000, seed 0); stem-label permutation p (10000, seed 0); Holm within the material family and separately within the covariate family.
2. Mahalanobis: standardize by reference median/MAD; Ledoit-Wolf covariance on reference stems; D² per incoming stem; null = leave-one-reference-stem-out D²; report fraction of incoming stems above null 95th pct. Run on material features and on all features.
3. MMD²: unbiased, RBF kernel with median-heuristic bandwidth on standardized stem vectors; stem-label permutation p (10000). Also tile-level MMD with stem-block permutation (permute stem→batch labels, tiles travel with stem).
4. PCA (and UMAP if installed) — visualization only, never a decision input.
Controls (required): (a) negative control: every 3-vs-4 split of Batch_1 stems as pseudo-batches → false-flag rate of each method; (b) positive control: inject synthetic shifts into copies of reference stems (gaussian blur σ=1.5, brightness +10%, add synthetic dark elliptical voids raising void_fraction by +0.02, add synthetic dark line cracks) → detection rate per method.
Flag wording (no semantic labels until checkpoint 3): "different from reference" if (MMD p<0.05 or >50% incoming stems over Mahalanobis null 95th) AND ≥1 material KPI Holm p<0.05; "investigate" if any one of those criteria; else "within observed reference variation".
Confound check per flagged batch: covariate family Holm results; Spearman ρ between each changed material KPI and each covariate across all stems (|ρ|>0.6 → "acquisition artifact could explain"); repeat material tests on stems matched by detector_set.
Evidence: per changed KPI, 3 incoming tiles farthest from ref median + 1 typical ref tile → PNG panel with KPI values.
Outputs: `$SEM_WORK_ROOT/drift/<run>/stem_features.csv`, `kpi_shifts.csv`, `mahalanobis.csv`, `mmd.json`, `controls.json`, `evidence/*.png`, `pca.png`, `report.md`. FORBIDDEN: training a classifier to predict batch ID as evidence of defects.

## 6. Conventions
- Branches: `devin/$(date +%s)-<slug>` off `devin/*-qc-foundation` (exact name in your brief). PR base = the foundation branch. NEVER merge.
- Configs: `configs/qc.yaml` is the single source of thresholds/paths/seeds; don't hard-code duplicates.
- Deps: add to `requirements.txt` (pinned `==`). Torch CPU wheels need `--extra-index-url https://download.pytorch.org/whl/cpu`.
- Every module has tests; run `pytest -q` before pushing. Report failures/disagreements explicitly; never drop or average away a result silently.
