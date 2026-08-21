# DPLM — a Dynamics-aware Protein Language Model

DPLM learns a joint embedding space in which a protein's **sequence** representation is
aligned with the **dynamics** of that protein observed in molecular-dynamics simulation.
An ESM2-650M encoder fitted with Houlsby adapters is trained contrastively (CLIP-style
InfoNCE) against pre-computed ViViT embeddings of ATLAS MD trajectories, so the resulting
residue- and protein-level representations carry information about flexibility and
correlated motion that a sequence-only model does not expose.

The repository contains everything needed to train DPLM, to extract representations from a
trained checkpoint, and to reproduce the eight downstream evaluations reported in the paper:
per-residue flexibility (RMSF), dynamic cross-correlation (DCCM), liquid–liquid phase
separation, cryptic binding-site prediction, three thermodynamic-stability tasks
(Mega-scale ΔΔG, S669, de novo designed proteins) and viral deep-mutational-scanning fitness.

```
train.py  infer.py  model.py           training, inference, architecture
configs/config_dplm.yaml               the published training configuration
data/  utils/  esm_adapterH/           dataloaders, helpers, ESM2-with-adapters fork
evaluate/
  methodology/   RMSF, adapter ablation, UMAP of the MD embedding space
  DCCM_v2/       dynamic cross-correlation: unsupervised + supervised, attention read-outs
  Phase_sep/     phase separation: XGBoost head + unsupervised t-SNE/K-means clustering
  cryptobench/   cryptic binding sites (XGBoost head)
  ddg_mega/      Mega-scale ΔΔG (linear head) and the de novo designed subset
  ddg_S669/      S669 ΔΔG (site-aware adapter head)
  fitness/       ProteinGym viral DMS, zero-shot
```

---

## 1. Installation

```bash
conda env create -f environment.yml
conda activate dplm_env
```

The environment mirrors the one used for every published number: **Python 3.10, PyTorch
2.4.0 (cu124), transformers 4.57.1, fair-esm 2.0.0, accelerate 0.24.1**. On a host with a
different CUDA version install PyTorch from <https://pytorch.org> first, then
`pip install -r requirements.txt`.

A CUDA GPU is required for training and for any task that runs the encoder; the XGBoost and
linear heads run on CPU once embeddings are cached. ESM2-650M weights (~2.5 GB) are
downloaded automatically by `fair-esm` on first use — set `TORCH_HOME` to a writable
directory if `$HOME` is quota-limited.

---

## 2. Training DPLM

Edit the six `Atlas_*_path` entries in `configs/config_dplm.yaml` to point at your copy of
the pre-processed ATLAS data (see §6 for what those directories contain), then:

```bash
accelerate launch train.py \
    --config_path ./configs/config_dplm.yaml \
    --result_path ./results/dplm
```

The published configuration is ESM2-650M with Houlsby adapters on the last 10 layers, Adam
at lr 7e-5, batch 32, fixed temperature 0.1, 30 000 steps. Validation runs every 50 steps and
three checkpoints are written to `results/dplm/checkpoints/`:

| checkpoint | selected by |
|---|---|
| `checkpoint_best_val_rmsf_cor.pth` | best validation RMSF correlation — **use this one** |
| `checkpoint_best_val_whole_loss.pth` | best validation contrastive loss |
| `checkpoint_every_n.pth` | rolling, every 500 steps (for resuming) |

> The validation RMSF correlation peaks early — typically within the first few hundred steps
> — and then declines, so `checkpoint_best_val_rmsf_cor.pth` is usually an early checkpoint.
> Training longer does not improve it.

Resume an interrupted run with `--resume_path <ckpt> --restart_optimizer 0`.

---

## 3. Pretrained checkpoint

The released DPLM checkpoint is hosted on HuggingFace, together with the exact training
configuration it was produced with:

<!-- TODO: replace with the real repo id once uploaded -->
**`https://huggingface.co/<ORG>/<MODEL_REPO>`**

```bash
pip install huggingface_hub
huggingface-cli download <ORG>/<MODEL_REPO> --local-dir ./checkpoints
# -> ./checkpoints/checkpoint_best_val_rmsf_cor.pth   (~3.2 GB)
# -> ./checkpoints/config_dplm.yaml                   (the config used to train it)
```

or in Python:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download('<ORG>/<MODEL_REPO>', 'checkpoint_best_val_rmsf_cor.pth')
cfg  = hf_hub_download('<ORG>/<MODEL_REPO>', 'config_dplm.yaml')
```

Use the `config_dplm.yaml` shipped **with the checkpoint** for inference and for the
downstream tasks — it records the architecture the weights were trained under (10 adapter
layers). The copy in `configs/` is identical apart from the data paths, which you edit for
training.

```bash
export CKPT=./checkpoints/checkpoint_best_val_rmsf_cor.pth
export CFG=./checkpoints/config_dplm.yaml
```

## 4. Extracting protein representations

`infer.py` returns the **pre-projection 1280-d DPLM representation** — the output of the
DPLM sequence encoder (ESM2 backbone + the contrastively-trained Houlsby adapters), taken
before the contrastive projection head. This is the representation every downstream task uses.

```bash
# protein level: mean-pooled, one vector per sequence -> [N, 1280]
python infer.py \
    --checkpoint_path results/dplm/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --input proteins.fasta --output embeddings.npy

# residue level: one matrix per sequence, true length preserved -> {id: [L, 1280]}
python infer.py \
    --checkpoint_path results/dplm/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --input proteins.fasta --output embeddings.npz --mode residue
```

In Python — the complete call sequence `infer.py` itself uses:

```python
import torch, yaml, numpy as np
from utils.utils import load_configs, load_dplm_checkpoint
from infer import build_dplm_model, embed_sequences

CFG  = './checkpoints/config_dplm.yaml'                       # from HuggingFace, see §3
CKPT = './checkpoints/checkpoint_best_val_rmsf_cor.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = load_configs(yaml.full_load(open(CFG)), args=None)
model, alphabet = build_dplm_model(configs, device)   # architecture: ESM2-650M + adapters
load_dplm_checkpoint(model, CKPT)                     # trained DPLM weights
model.eval()

seqs = [('protein_A', 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR'),
        ('protein_B', 'GSHMSLQDPFLNALRRERVPVSIYLVNGIKLQGQIESFDQFVILLKNTVSQMVYKHAISTVVPS')]

# protein level: mean-pooled over residues -> [N, 1280]
_, X = embed_sequences(model, alphabet, seqs, batch_size=8,
                       max_length=configs.model.esm_encoder.max_length,
                       device=device, mode='protein')
print(X.shape)                                                 # (2, 1280)

# residue level: one [L_i, 1280] array per sequence, true length, no padding
ids, R = embed_sequences(model, alphabet, seqs, batch_size=8,
                         max_length=configs.model.esm_encoder.max_length,
                         device=device, mode='residue')
for i, r in zip(ids, R):
    print(i, r.shape)                                          # protein_A (78, 1280), protein_B (64, 1280)
```

These 1280-d vectors are the input every task head below expects.

## 5. Predicting on new proteins with the trained task heads

Each task ships a fitted head that consumes DPLM representations. Download the heads
alongside the checkpoint (§3); the snippets below assume `model`, `alphabet`, `configs` and
`device` from §4 are already in scope, and `MAXLEN = configs.model.esm_encoder.max_length`.

> All three heads were fitted on representations from
> `checkpoint_best_val_rmsf_cor.pth`. Using a different DPLM checkpoint changes the input
> distribution and the heads will be miscalibrated — re-fit them if you retrain DPLM.

### 5.1 Phase separation — protein level, XGBoost

Model: `Phase_sep/xgboost/xgb_phase_sep.pkl` (a pickled `XGBClassifier`).
Input: one **mean-pooled** 1280-d vector per protein.

```python
import pickle, numpy as np
from infer import embed_sequences

xgb = pickle.load(open('Phase_sep/xgboost/xgb_phase_sep.pkl', 'rb'))

seqs = [('my_protein', 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDG...')]
_, X = embed_sequences(model, alphabet, seqs, batch_size=8,
                       max_length=MAXLEN, device=device, mode='protein')   # [N, 1280]

prob = xgb.predict_proba(X)[:, 1]                 # P(phase-separating)
for (pid, _), p in zip(seqs, prob):
    print(f'{pid}: P(LLPS) = {p:.3f}  ->  {"phase-separating" if p >= 0.5 else "not"}')
```

### 5.2 Cryptic binding sites — per residue, XGBoost

Model: `cryptobench_dplm/xgb_pred/checkpoints/xgb_model.json` plus `threshold.json`
(the F1-optimal cut, **0.5975** — not 0.5, because the task is ~19:1 imbalanced).
Input: the **per-residue** 1280-d vectors; each residue is scored independently.

```python
import json, xgboost as xgb, numpy as np
from infer import embed_sequences

booster = xgb.Booster(); booster.load_model('cryptobench_dplm/xgb_pred/checkpoints/xgb_model.json')
thr = json.load(open('cryptobench_dplm/xgb_pred/checkpoints/threshold.json'))['threshold']

seq = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR'
ids, R = embed_sequences(model, alphabet, [('my_protein', seq)], batch_size=1,
                         max_length=MAXLEN, device=device, mode='residue')
emb = R[0]                                        # [L, 1280], one row per residue

prob = booster.predict(xgb.DMatrix(emb))          # [L] P(cryptic site) per residue
site = np.where(prob >= thr)[0]
print('predicted cryptic-site residues (0-indexed):', site.tolist())
for i in site:
    print(f'  {i+1:4d} {seq[i]}  p={prob[i]:.3f}')
```

### 5.3 ΔΔG of a point mutation — site-aware adapter head

Model: `ddg_adam_v9_sw3/checkpoints/best_model_cor.pth` (2.9 GB — it contains its own
ESM2+adapter encoder, so it does **not** reuse the `model` from §4) with
`ddg_config_siteaware_v3.yaml`. It predicts ΔΔG directly from the wild-type sequence, the
mutant sequence and the 0-indexed mutated position.

```python
import sys, torch, yaml, logging
from box import Box
sys.path += ['evaluate/ddg_S669']
from model_ddg_v2 import prepare_models_v2

# NOTE: the ddG head uses its OWN config schema (optimizer.lr, not the training config's
# optimizer.lr_seq), so utils.utils.load_configs does NOT apply here — Box-wrap the YAML
# directly, exactly as train_ddg_v2._load_configs does.
cfg = Box(yaml.full_load(open('ddg_adam_v9_sw3/ddg_config_siteaware_v3.yaml')))
cfg.encoder.adapter_h.num_end_adapter_layers = [10, 4]      # must match the trained head
cfg.train_settings.device = str(device)      # the YAML hardcodes 'cuda'; override for CPU
net = prepare_models_v2(cfg, logging)        # needs a logger, not None
ckpt = torch.load('ddg_adam_v9_sw3/checkpoints/best_model_cor.pth', map_location='cpu')
net.load_state_dict(ckpt['model_state_dict'])
net.eval().to(device)

wt  = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR'
pos = 10                                    # 0-indexed site of the mutation
mt  = wt[:pos] + 'A' + wt[pos+1:]           # Q11A in 1-based notation

# forward(from_seqs, to_seqs, mut_pos): three PARALLEL LISTS, one entry per sample.
# mut_pos is a list *of lists* — the 0-indexed sites where the two sequences differ, so a
# multi-point mutant is [[3, 27, 40]]. It is a plain Python list, not a tensor.
with torch.no_grad():
    ddg = net([wt], [mt], [[pos]])          # -> tensor of shape [B]
print(f'predicted ddG = {float(ddg[0]):.3f} kcal/mol '
      f'({"destabilising" if float(ddg[0]) > 0 else "stabilising"})')
```

Sign convention follows the S669 training data: **positive ΔΔG = destabilising**.

### Where the trained heads live on LCC

| task | head |
|---|---|
| phase separation | `$V/Phase_sep/xgboost/xgb_phase_sep.pkl` |
| cryptic binding site | `$V/cryptobench_dplm/xgb_pred/checkpoints/{xgb_model.json,threshold.json}` |
| ΔΔG (site-aware) | `results/ddg_adam_v9_sw3/checkpoints/best_model_cor.pth` + its `ddg_config_siteaware_v3.yaml` |

with `V=$DATA/old_results/comparable/lcc_adam_v9`.

## 6. Reproducing the evaluation results

Set the two paths once. `$DATA` is the dataset root (§6); `$CKPT` is the released DPLM
checkpoint.

```bash
export DATA=/path/to/DPLM_data          # where you downloaded the datasets (see below)
export CKPT=./checkpoints/checkpoint_best_val_rmsf_cor.pth
export CFG=./checkpoints/config_dplm.yaml
export PYTHONPATH=$(pwd):$PYTHONPATH
```

```bash
# 1. RMSF — per-residue flexibility, ATLAS held-out proteins
python evaluate/methodology/rmsf.py --methods dplm \
    --data_path $DATA/processed_test_rep2 --analysis_path $DATA/analysis \
    --dplm_config $CFG --dplm_checkpoint $CKPT \
    --metric rmsf --rmsf_col RMSF_R2 --output_dir ./results/rmsf

# 2. Adapter ablation — trained vs randomised adapters vs base ESM2
python evaluate/methodology/ablation.py \
    --data_path $DATA/processed_test_rep2 --analysis_path $DATA/analysis \
    --dplm_config $CFG --dplm_checkpoint $CKPT \
    --random_mode shuffle --random_seeds 0+1+2 --output_dir ./results/ablation

# 3. UMAP of the MD embedding space
python evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir $DATA/processed_data_rep0 --checkpoint $CKPT --config_path $CFG \
    --seq_model_location $CKPT --output_dir ./results/md_emb_umap

# 4. DCCM, unsupervised — embedding-similarity read-out
python evaluate/DCCM_v2/unsup_dccm.py --methods dplm \
    --data_path $DATA/processed_data_rep2 --analysis_path $DATA/analysis \
    --dccm_dir $DATA/DCCM_dir3/ --dplm_config $CFG --dplm_checkpoint $CKPT \
    --output_dir ./results/dccm_unsup_emb

# 5. DCCM, unsupervised — attention read-out
python evaluate/DCCM_v2/unsup_dccm_attn_v2.py --methods dplm \
    --data_path $DATA/processed_data_rep2 --analysis_path $DATA/analysis \
    --dccm_dir $DATA/DCCM_dir3/ --dplm_config $CFG --dplm_checkpoint $CKPT \
    --output_dir ./results/dccm_unsup_attn

# 6. DCCM, supervised — channel-matched attention predictor, 3 seeds
python evaluate/DCCM_v2/train_dccm_attn_v2.py --methods dplm \
    --train_data_path $DATA/processed_data_rep2 --test_data_path $DATA/processed_test_rep2 \
    --analysis_path $DATA/analysis --dccm_dir $DATA/DCCM_dir3/ \
    --dplm_config $CFG --dplm_checkpoint $CKPT \
    --num_attn_layers 10 --seeds 0+1+2 --output_dir ./results/dccm_sup

# 7. Phase separation
python evaluate/Phase_sep/phase_separation_xgboost.py \
    --config_path $CFG --checkpoint_path $CKPT \
    --train_pos $DATA/Phase_sep/Molphase_train_pos.xlsx \
    --train_neg $DATA/Phase_sep/Molphase_train_neg.xlsx \
    --test_dir $DATA/Phase_sep --output_path ./results/phase_sep/ --batch_size 1

# 8. Cryptic binding sites
python evaluate/cryptobench/embed_dplm.py \
    --dataset_dir $DATA/cryptobench/cryptobench-dataset \
    --cif_dir $DATA/cryptobench/cryptobench-dataset/auxiliary-data/cif-files \
    --output_dir ./cryptobench_emb --config_path $CFG --checkpoint $CKPT \
    --batch_size 4 --skip_existing
python evaluate/cryptobench/train_cryptobench_xgb.py \
    --config_path evaluate/cryptobench/cryptobench_config.yaml \
    --emb_dir ./cryptobench_emb --dataset_dir $DATA/cryptobench/cryptobench-dataset \
    --result_path ./results/cryptobench --save_predictions

# 9. ΔΔG Mega-scale (linear head)
python evaluate/ddg_mega/ddg_mega_scale.py \
    --csv_path $DATA/ddg/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --model_type d-plm --config_path $CFG --checkpoint_path $CKPT \
    --output_dir ./results/ddg_mega --save_predictions

# 10. ΔΔG on the de novo designed subset — supervised and zero-shot
python evaluate/ddg_mega/ddg_designed.py --mode supervised --method dplm \
    --csv_path $DATA/ddg_designed/tsuboyama_designed146_mutations.csv \
    --dplm_config $CFG --dplm_checkpoint $CKPT \
    --output_dir ./results/designed_supervised --save_predictions
python evaluate/ddg_mega/ddg_designed.py --mode zeroshot --method dplm \
    --csv_path $DATA/ddg_designed/tsuboyama_designed146_mutations.csv \
    --dplm_config $CFG --dplm_checkpoint $CKPT \
    --output_dir ./results/designed_zeroshot --save_predictions

# 11. ΔΔG S669
accelerate launch evaluate/ddg_S669/train_ddg_v2.py \
    --config_path evaluate/ddg_S669/ddg_config_siteaware_v3.yaml \
    --result_path ./results/ddg_S669 --resume_path $CKPT \
    --train_csv_path $DATA/ddg/S8754.csv --test_csv_path $DATA/ddg/S669.csv \
    --num_end_adapter_layers 10,4 --save_predictions

# 8b. Phase separation, UNSUPERVISED — t-SNE + K-means(k=2) clustering, ARI vs labels
python evaluate/Phase_sep/phase_sep_viz.py --model_type d-plm \
    --checkpoint_path $CKPT --config_path $CFG \
    --data_dir $DATA/Phase_sep --output_dir ./results/phase_sep_viz \
    --tables S1 S2 S3 S4 S5 --batch_size 8 --save_emb

# 12. Viral DMS fitness, zero-shot (wt-mt RLA)
python evaluate/fitness/predict_fitness_viral.py --methods dplm \
    --manifest $DATA/proteingym/viral31_manifest.csv --max_seq_len 1024 \
    --dms_dir $DATA/proteingym/DMS_ProteinGym_substitutions \
    --dplm_config $CFG --dplm_checkpoint $CKPT --output_dir ./results/viral
python evaluate/fitness/collect_viral_results.py --results_dir ./results/viral
```

Reference outputs for all of the above, produced with the released checkpoint on the **full
(unfiltered)** datasets, are published alongside the checkpoint — subdirectories `rmsf/`,
`ablation/`, `MD_emb_eval/`, `Phase_sep/`, `cryptobench_dplm/`, `ddg_meg_adam_v9/`,
`ddg_S669/` and `fitness41/`.

Notes that matter for exact reproduction:

* **DCCM tasks read `processed_data_rep2`**, which is the ATLAS *training* partition. That is
  how the published numbers were produced. Pass `--data_path $DATA/processed_test_rep2` to
  score held-out proteins instead.
* **RMSF replicates.** `--rmsf_col` selects `RMSF_R1`, `RMSF_R2` (default) or `RMSF_R3`. The
  choice moves the mean correlation by less than 0.01.
* **The designed subset is 146 proteins, not 156.** An earlier extraction called a protein
  designed whenever its `WT_name` did not start with a 4-character PDB id, which wrongly
  included 10 `v2*` entries that embed real PDB ids and are natural domains. Excluding them
  reproduces the source paper's 331 natural / 148 designed split (146 survive the ddG/indel
  quality filter). `designed146_protein_list.csv` is the canonical list.
* **XGBoost and linear heads are deterministic**; the supervised DCCM head is seeded via
  `--seeds`. Contrastive training itself is seeded through `fix_seed` in the config, but
  run-to-run variation in validation RMSF correlation is still substantial — compare
  configurations across several seeds, never from single runs.

---

## 7. Datasets

All paths below are relative to `$DATA`, the directory you download the datasets into.
Set `export DATA=/path/to/DPLM_data` and use it consistently; the training config expects
the same layout (§2).

| purpose | path | size |
|---|---|---|
| ATLAS MD embeddings, training split (3 replicates) | `processed_data_rep{0,1,2}/` | 1.5 G each |
| ATLAS MD embeddings, held-out split (3 replicates) | `processed_test_rep{0,1,2}/` | 74 M each |
| ATLAS structures + per-residue RMSF/Bfactor/Neq tables | `analysis/{pid}_analysis/` | 111 G |
| Ground-truth DCCM matrices | `DCCM_dir3/` | 3.6 G |
| Phase separation (Molphase train + test tables S1–S5) | `Phase_sep/` | 3.6 M |
| CryptoBench dataset + mmCIF structures | `cryptobench/cryptobench-dataset/` | 6.5 G |
| Mega-scale ΔΔG (Tsuboyama 2023) | `ddg/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` | 666 M |
| ΔΔG S669 test / S8754 train | `ddg/S669.csv`, `ddg/S8754.csv` | 512 K / 3.0 M |
| De novo designed ΔΔG subset (146 proteins) | `ddg_designed/tsuboyama_designed146_mutations.csv` | 75 M |
| ProteinGym viral manifests | `proteingym/viral{23,31}_manifest.csv` | 512 K |
| ProteinGym DMS assay tables | `proteingym/DMS_ProteinGym_substitutions/` | 1.1 G |
| **Released DPLM checkpoint** | HuggingFace, see §3 — *not* under `$DATA` | 3.2 G |

Original sources: ATLAS (<https://www.dsimb.inserm.fr/ATLAS>), CryptoBench, Tsuboyama et al.
2023 Mega-scale stability, S669/S8754, MolPhase, and ProteinGym v1.2.
