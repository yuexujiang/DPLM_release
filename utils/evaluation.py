import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from Bio.PDB import PDBParser, PPBuilder


def compute_repdiff(row, sequence, model, alphabet, offset_idx, wt_representation, mode="whole", similarity="correlation"):
    if ":" not in row:
        base = row[1:-1]
        if len(base) == 0:
            return np.nan

        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        if idx >= len(sequence):
            return np.nan

        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        sequence = sequence[:idx] + mt + sequence[(idx + 1):]
    else:
        for row_i in row.split(":"):
            base = row[1:-1]
            if len(base) == 0:
                return np.nan

            wt, idx, mt = row_i[0], int(row_i[1:-1]) - offset_idx, row_i[-1]
            if idx >= len(sequence):
                return np.nan

            assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            sequence = sequence[:idx] + mt + sequence[(idx + 1):]

    if "_" in sequence:
        return np.nan

    data = [("protein1", sequence)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    mt_representation = model(batch_tokens.cuda(), repr_layers=[model.num_layers])["representations"][model.num_layers].squeeze(0)

    if mode == "whole":
        mt_representation = torch.sum(mt_representation, dim=0)
        wt_representation = torch.sum(wt_representation, dim=0)
    elif mode == "marginals":
        mt_representation = mt_representation[idx + 1:idx + 2].squeeze(0)
        wt_representation = wt_representation[idx + 1:idx + 2].squeeze(0)
    elif mode == "RLA":
        if similarity == "cosine":
            score = (mt_representation.unsqueeze(0).unsqueeze(2) @ wt_representation.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(-1).squeeze(0)
            return (score).mean(0).item()
        elif similarity == "euclidean_distance":
            score = np.linalg.norm((mt_representation - wt_representation).to('cpu').detach().numpy(), axis=1)
            return np.log(np.mean(score)) * -1
        elif similarity == "mse":
            score = np.mean(((mt_representation - wt_representation).to('cpu').detach().numpy()) ** 2)
            return np.log(score) * -1

    if similarity == "cosine":
        score = F.cosine_similarity(mt_representation.unsqueeze(0), wt_representation.unsqueeze(0))
    elif similarity == "euclidean_distance":
        score = torch.dist(mt_representation, wt_representation, p=2)
    elif similarity == "correlation":
        stacked_tensors = torch.stack((mt_representation, wt_representation))
        correlation_matrix = torch.corrcoef(stacked_tensors)
        score = correlation_matrix[0, 1]

    return np.log(score.item())


def test_DMS(configs, seq_model, alphabet, n_steps, logging):
    outputfilename = os.path.join(configs.valid_settings.dms_summary_path)
    dfdata = pd.read_csv(outputfilename)
    tqdm.pandas()
    for index, row in dfdata.iterrows():
        wt_seq = row['WT seq']
        if wt_seq is np.nan:
            continue

        filekey = row['Dataset_file']
        if not filekey == 'PTEN_HUMAN_Fowler2018':
            continue

        path_test = os.path.join(configs.valid_settings.dms_path, 'test', row['Dataset_file'] + ".csv")
        path_val = os.path.join(configs.valid_settings.dms_path, 'validation', row['Dataset_file'] + ".csv")
        if os.path.exists(path_test):
            dms_input = path_test
        else:
            dms_input = path_val

        sequence = str(wt_seq)
        offset_idx = int(row['offset_idx'])
        ref_name = row['Name(s) in Reference']

        df = pd.read_csv(dms_input)
        batch_converter = alphabet.get_batch_converter()
        data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        mode = "RLA"
        tqdm.pandas()
        with torch.no_grad():
            wt_representation = seq_model(batch_tokens.cuda(), repr_layers=[seq_model.num_layers])["representations"][seq_model.num_layers]

        wt_representation = wt_representation.squeeze(0)
        mutation_col = "mutant"
        similarity = "euclidean_distance"
        df['modelname'] = df.progress_apply(
            lambda row: compute_repdiff(
                row[mutation_col],
                sequence,
                seq_model,
                alphabet,
                offset_idx,
                wt_representation,
                mode=mode,
                similarity=similarity
            ),
            axis=1,
        )
        esm_rla_spearmn = df[ref_name].corr(df['modelname'], method='spearman')

    logging.info(f"step:{n_steps} esm_rla_spearmn:{esm_rla_spearmn:.4f}")
    return esm_rla_spearmn


def pdb2seq(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            for poly_index, poly in enumerate(polypeptides):
                sequence = poly.get_sequence()
    return sequence


def test_rmsf_cor(val_loader, alphabet, configs, seq_model, n_steps, logging, replicate):
    processed_list = []
    result_list = []
    for batch in val_loader:
        rep_norm_list = []
        rmsf_list = []
        pid = batch['pid'][0]
        pid = pid.split('#')[0]
        if pid in processed_list:
            continue
        pdb_file = os.path.join(configs.valid_settings.analysis_path, f"{pid}_analysis", f"{pid}.pdb")
        sequence = str(pdb2seq(pdb_file))
        data = [("protein1", sequence)]
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            wt_representation = seq_model(batch_tokens.cuda(), repr_layers=[seq_model.num_layers])["representations"][seq_model.num_layers]

        wt_representation = wt_representation.squeeze(0)
        seq_emb = wt_representation[1:-1]
        residue_norms = np.linalg.norm(seq_emb.cpu(), axis=1)
        rmsf_file = os.path.join(configs.valid_settings.analysis_path, f"{pid}_analysis", f"{pid}_RMSF.tsv")
        df = pd.read_csv(rmsf_file, sep="\t")
        rmsf_col_name = "RMSF_R" + str(int(replicate) + 1)
        r1 = df[rmsf_col_name].values
        rep_norm_list.extend(residue_norms)
        rmsf_list.extend(r1)
        processed_list.append(pid)
        corr, _ = spearmanr(rep_norm_list, rmsf_list)
        result_list.append(corr)

    result_mean = np.mean(result_list)
    logging.info(f"step:{n_steps} rmsf_cor:{result_mean:.4f}")
    return result_mean
