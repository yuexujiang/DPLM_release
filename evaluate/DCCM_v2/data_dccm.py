"""
data_dccm.py — data assembly for the DCCM-prediction downstream task.

Reuses the existing methodology code:
  * protein enumeration + ground-truth DCCM   → evaluate/methodology/protein_level_emb_md.py
  * per-residue embedding extraction / models  → evaluate/methodology/rmsf.py

A "sample" is one protein: {pid, emb [L, D] (np.float32), dccm [L, L] (np.float32)}.
Train proteins come from --train_data_path (processed_data), test proteins from
--test_data_path (processed_test); ground-truth DCCM matrices are computed from the MD
trajectories under --analysis_path ({pid}_analysis/{pid}.pdb + {pid}_R{1,2,3}.xtc).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methodology'))

from rmsf import (load_esm2, load_esmc, load_prostt5, load_seqdance, load_dplm,  # noqa: E402
                  load_splm, build_embedding_cache)
from protein_level_emb_md import load_proteins, load_protein_dccm           # noqa: E402


# Map the CLI method name to the key build_embedding_cache dispatches on.
# NOTE: 'DPLM' and 'ESM2' both go through build_embedding_cache's ESM2-style path.
METHOD_KEY = {'esm2': 'ESM2', 'esmc': 'ESMC', 'prostt5': 'ProstT5',
              'seqdance': 'SeqDance', 'dplm': 'DPLM', 'splm': 'SPLM'}


def load_method_model(method, device, seqdance_path=None,
                      dplm_config=None, dplm_checkpoint=None,
                      proteins=None, splm_path=None, splm_config=None,
                      splm_checkpoint=None, splm_python=None,
                      splm_max_length=1022, splm_cache_pkl=None,
                      esmc_model='esmc_600m'):
    """Return a single-entry models_dict {method_key: (model, extra)} for the chosen method.

    esm2     : frozen vanilla ESM2-650M.
    esmc     : frozen vanilla ESM-C (default 600M / 1152-d; see esmc_local/).
    prostt5  : frozen ProstT5 encoder.
    seqdance : frozen SeqDance/ESMDance (needs seqdance_path).
    dplm     : ESM2 + Houlsby adapters loaded from a DPLM training checkpoint
               (needs dplm_config + dplm_checkpoint). The backbone is frozen; only the
               downstream DCCM head is trained, same as the other methods.
    splm     : SPLM-V2-GVP, run OUT-OF-PROCESS (needs splm_path/config/checkpoint and,
               ideally, splm_python pointing at the splm_v2 env). load_splm precomputes a
               {sequence: [L, D]} lookup for every protein, so `proteins` must cover ALL
               sequences that will later be embedded (train + test combined).
    """
    method = method.lower()
    if method == 'esm2':
        model, extra = load_esm2(device)
    elif method == 'esmc':
        model, extra = load_esmc(device, esmc_model)
    elif method == 'prostt5':
        model, extra = load_prostt5(device)
    elif method == 'seqdance':
        if not seqdance_path:
            raise ValueError('seqdance requires --seqdance_path')
        model, extra = load_seqdance(seqdance_path, device)
    elif method == 'dplm':
        if not (dplm_config and dplm_checkpoint):
            raise ValueError('dplm requires --dplm_config and --dplm_checkpoint')
        model, extra = load_dplm(dplm_config, dplm_checkpoint, device)
    elif method == 'splm':
        if not (splm_path and splm_config and splm_checkpoint):
            raise ValueError('splm requires --splm_path, --splm_config and --splm_checkpoint')
        if proteins is None:
            raise ValueError('splm requires the full protein list (train + test)')
        model, extra = load_splm(
            proteins, splm_path, splm_config, splm_checkpoint, device,
            splm_python=splm_python, max_length=splm_max_length,
            cache_pkl=splm_cache_pkl)
    else:
        raise ValueError(
            f'Unknown method: {method} '
            f'(expected esm2 | esmc | prostt5 | seqdance | dplm | splm)')
    return {METHOD_KEY[method]: (model, extra)}


def build_samples(proteins, models_dict, analysis_path, device,
                  dccm_dir=None, replicate=None):
    """Build a list of {pid, emb, dccm} samples for the given proteins + one model.

    Uses build_embedding_cache (which drops proteins with an embedding/length mismatch),
    then attaches the ground-truth DCCM; proteins whose DCCM is missing or shape-mismatched
    are skipped.
    """
    method_key = next(iter(models_dict))
    cache, valid_proteins = build_embedding_cache(proteins, models_dict, device)

    samples = []
    for prot in valid_proteins:
        pid = prot['pid']
        emb = cache[pid][method_key]                       # [L, D]
        dccm = load_protein_dccm(pid, analysis_path,
                                 dccm_dir=dccm_dir, replicate=replicate)
        if dccm is None:
            print(f'  [skip] {pid}: ground-truth DCCM unavailable')
            continue
        if dccm.shape != (emb.shape[0], emb.shape[0]):
            print(f'  [skip] {pid}: DCCM shape {dccm.shape} != emb len {emb.shape[0]}')
            continue
        samples.append({
            'pid':  pid,
            'emb':  emb.astype(np.float32),
            'dccm': dccm.astype(np.float32),
        })

    print(f'  built {len(samples)} samples (from {len(valid_proteins)} embedded proteins)')
    return samples
