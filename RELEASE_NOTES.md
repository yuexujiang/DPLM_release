# Release notes

This tree is a clean extraction of the DPLM research repository, containing only the code
needed to train DPLM, extract representations, and reproduce the eight DPLM evaluations.
Baseline encoders (ESM2, ProstT5, SeqDance/ESMDance, ESM-C, SPLM), combined-feature variants,
the MDGen and SPLM training arms, cluster job scripts, and the allostery analyses were all
removed.

## Two changes were made to the extracted code

Both are release-only; the research repository is untouched.

1. **`evaluate/DCCM_v2/unsup_dccm_attn_v2.py`** — an unescaped `%` in an argparse help string
   made `--help` raise. Escaped to `%%`. No effect on behaviour.

2. **`evaluate/ddg_mega/ddg_mega_scale_baseline.py`** — `from splm_embed import encode_splm`
   was a top-level import. SPLM is a baseline and is not shipped, so the import now degrades
   to a stub that raises only if `--method splm` is actually selected. This module is present
   because `ddg_designed.py` imports `evaluate_proteins()` from it for the **DPLM** path.

## Superseded modules were merged away

Five pre-v2 modules were deleted and the functions still used from them inlined into their
successors, each under a comment naming its origin:

| removed (lines) | symbols moved | inlined into |
|---|---|---|
| `DCCM_v2/train_dccm_attn.py` (312) | `_to_tensors`, `dccm_loss`, `dccm_corr_loss`, `evaluate` | `train_dccm_attn_v2.py` |
| `DCCM_v2/plot_dccm.py` (141) | `draw_dccm_pair`, `plot_corr_comparison`, `_pval_str`, `METHOD_COLORS` | `unsup_dccm.py` |
| `ddg_mega/ddg_mega_scale_baseline.py` (288) | `evaluate_proteins`, `compute_features` | `ddg_designed.py` |
| `ddg_S669/model_ddg.py` (346) | `prepare_adapter_h_model`, `prepare_esm_model`, `print_trainable_parameters`, +2 helpers | `model_ddg_v2.py` |
| `ddg_S669/data_ddg.py` (103) | `extract_protein_id` | `data_ddg_v2.py` |

**1190 lines, 5 files.** The criterion for absorbing a module is (a) exactly one consumer and
(b) a dependency closure that resolves in the target — not how much of the file is used.

The merge tool resolves **every free name** in each moved function against the target module's
namespace and refuses to write if any is unresolved. It blocked three merges that would
otherwise have raised `NameError` at call time rather than at import:

* `dccm_loss` / `dccm_corr_loss` needed `dccm_mse_loss` and `_upper_tri_mask` from `model_dccm`;
* `evaluate_proteins` needed `LinearRegression`, `mean_absolute_error`, `tqdm`, `parse_mut_type`;
* `plot_corr_comparison` read `METHOD_COLORS`, a module-level constant.

The missing imports and the constant were added to the targets.

## Files kept because they have MORE THAN ONE consumer

These are genuinely shared libraries, not superseded versions:

* `evaluate/DCCM_v2/{data_dccm,data_dccm_attn,model_dccm,model_dccm_attn}.py` — the DCCM data
  and model layer, used by both the unsupervised and the supervised v2 scripts.
* `evaluate/methodology/protein_level_emb_md.py` — `load_proteins()` / `load_protein_dccm()`.
* `evaluate/methodology/rmsf.py` — also imported by `ablation.py` and `predict_fitness_viral.py`.

## `phase_sep_viz.py` was trimmed to DPLM-only

The unsupervised arm of the phase-separation task (t-SNE + K-means k=2 + ARI on tableS1-S5)
originally loaded ESM2 alongside DPLM and carried branches for ESM-C, ProstT5, SPLM and
SeqDance. Removed: the ESM2 co-load, all non-DPLM branches, `_load_esm2` / `_load_seqdance` /
`_encode_prostt5` / `_encode_seqdance`, the top-level `from splm_embed import encode_splm`,
the four dead `if <baseline> is not None:` blocks inside `analyze_table`, and 8 baseline CLI
flags. `--model_type` is now `choices=['d-plm']`. 604 -> 466 lines.

Because this file was edited rather than copied, it was validated by an end-to-end run on
tableS4 (36 sequences) with the released checkpoint: ARI = 0.5942, both figures written.

## Verified

All 14 entry points import and respond to `--help` inside this tree, and every `.py` file
parses, using the `dplm_env` environment described in `README.md`.
