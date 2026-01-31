DPLM: Dynamics-aware Protein Language Model

DPLM is a dynamics-aware protein language model that enriches protein sequence representations with information learned from molecular dynamics (MD) trajectories. DPLM is pretrained using unsupervised contrastive learning, aligning sequence embeddings with embeddings of real MD trajectories.
The resulting representations capture protein flexibility and dynamics while requiring only sequence input at inference time.

This repository provides code for:

Extracting DPLM protein representations

Running intrinsic disorder region (IDR) prediction

Running protein stability change (ΔΔG) prediction

🔗 Resources

Code repository: https://github.com/yuexujiang/DPLM_release

Pretrained checkpoints (Hugging Face):
https://huggingface.co/Yuexuhug/DPLM/tree/main
Installation

We recommend using a conda environment.

conda create -n dplm python=3.9 -y
conda activate dplm
pip install -r requirements.txt

Pretrained Checkpoints

Download the required checkpoints from Hugging Face and place them in the checkpoint/ directory.

Task	Checkpoint file
Representation extraction	checkpoint_best_val_rmsf_cor.pth
IDR prediction	idr.pth
ΔΔG prediction	ddt.pth
1️⃣ Extracting DPLM Protein Representations

DPLM produces per-sequence embeddings that encode dynamic information.

Example
from utils.utils import *

model_location = './checkpoint/checkpoint_best_val_rmsf_cor.pth'
model_config = './config/config_vivit3.yaml'

input_data = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]

model, alphabet = load_model(model_config, model_location)

embeddings = []
for seq in input_data:
    embeddings.append(extract_emb_perseq(model, alphabet, seq))


Each entry in embeddings is a dynamics-aware protein representation.

2️⃣ Intrinsic Disorder Region (IDR) Prediction

DPLM can be adapted to residue-level IDR prediction using a lightweight prediction head.

Example
from model_idr import load_model_idr

config_path = './config/idr_config_30CAID2_trainfix_adp16_adp4.yaml'
model_location = './checkpoint/idr.pth'

model = load_model_idr(config_path, model_location)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
result = model([sequence])


Output: a per-residue disorder probability.

3️⃣ Protein Stability Change (ΔΔG) Prediction

DPLM can also be adapted for mutation-induced stability change prediction.

Example
from model_ddt import load_model_ddt

config_path = './config/ddt_config_adapterH16_adapterH4.yaml'
model_location = './checkpoint/ddt.pth'

model = load_model_ddt(config_path, model_location)

wild_seq = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
]
mut_seq = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLKEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
]

result = model(wild_seq, mut_seq)


Output: predicted ΔΔG value for the mutation.