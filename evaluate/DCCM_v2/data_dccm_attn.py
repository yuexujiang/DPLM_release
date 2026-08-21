"""
data_dccm_attn.py — data assembly for the *attention-augmented* DCCM predictor.

This is the attention-map variant of data_dccm.py. In addition to the per-residue
embedding [L, D] it also extracts the backbone's self-attention maps and reduces them to a
compact pairwise tensor [C, L, L] that is fed directly to the pair head (model_dccm_attn.py).

Motivation: DCCM is an [L, L] pairwise coupling matrix, but a per-residue embedding only
lets the head *reconstruct* coupling via a bilinear form. Transformer attention maps are
natively pairwise and are the classic signal for residue–residue coupling / contacts. For
DPLM specifically, the Houlsby adapters reshape attention most strongly, so exposing the
attention maps gives the MD-aligned adapter training a direct path to the DCCM output.

Supported backbones (each via its own attention API, normalised to a common [layers, heads,
L, L] tensor before reduction):
  esm2, dplm : model(..., need_head_weights=True) → result["attentions"] [B, layers, heads, T, T]
  esmc       : ESMplusplusModel(..., output_attentions=True) → tuple(layers) of [B, heads, T, T]
               (output_attentions switches the block off SDPA onto the explicit-softmax path,
                so the returned weights are the true post-softmax attention)
  prostt5    : T5EncoderModel(..., output_attentions=True) → tuple(layers) of [B, heads, T, T]
  seqdance   : ESMwrap(..., return_attention_map=True) → ["attention_map"] [B, T, T, layers*heads]
               (concatenated layer-major, reshaped back to [layers, heads, L, L])

**SPLM is the one exception** — it runs out-of-process and returns per-residue embeddings only,
with no attention available. Use the embedding-only data_dccm.py / unsup_dccm.py for SPLM.

Because the reduction takes the last N layers and averages over heads, every backbone yields the
SAME channel count C = N regardless of its own layer/head geometry — so the attention feature is
directly comparable across methods.

A "sample" is one protein:
  {pid, emb [L, D] float32, attn [C, L, L] float16, dccm [L, L] float32}.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methodology'))

from rmsf import load_esm2, load_esmc, load_dplm, load_prostt5, load_seqdance   # noqa: E402
from protein_level_emb_md import load_proteins, load_protein_dccm           # noqa: E402


# Map the CLI method name to a display key. Attention maps are available for every backbone
# below; SPLM is excluded (out-of-process, embeddings only).
METHOD_KEY = {'esm2': 'ESM2', 'dplm': 'DPLM', 'esmc': 'ESMC',
              'prostt5': 'ProstT5', 'seqdance': 'SeqDance'}

# Backbones whose attention this module can extract.
ATTN_METHODS = set(METHOD_KEY)


def load_attn_model(method, device, dplm_config=None, dplm_checkpoint=None,
                    seqdance_path=None, esmc_model='esmc_600m'):
    """Load an attention-capable backbone → (model, extra).

    esm2     : frozen vanilla ESM2-650M                       (extra = alphabet)
    dplm     : ESM2 + Houlsby adapters from a DPLM checkpoint  (extra = alphabet)
    esmc     : frozen vanilla ESM-C (default 600M)             (extra = tokenizer)
    prostt5  : frozen ProstT5 encoder                          (extra = tokenizer)
    seqdance : frozen SeqDance/ESMDance                        (extra = tokenizer)

    The backbone is always frozen; only the downstream DCCM head trains.
    """
    method = method.lower()
    if method == 'esm2':
        return load_esm2(device)
    if method == 'esmc':
        return load_esmc(device, esmc_model)
    if method == 'dplm':
        if not (dplm_config and dplm_checkpoint):
            raise ValueError('dplm requires --dplm_config and --dplm_checkpoint')
        return load_dplm(dplm_config, dplm_checkpoint, device)
    if method == 'prostt5':
        return load_prostt5(device)
    if method == 'seqdance':
        if not seqdance_path:
            raise ValueError('seqdance requires --seqdance_path')
        return load_seqdance(seqdance_path, device)
    raise ValueError(
        f"Unknown / unsupported method for the attention variant: {method} "
        f"(supported: {sorted(ATTN_METHODS)}; SPLM has no attention — it runs "
        f"out-of-process and returns embeddings only)")


def _reduce_attn(attn, num_attn_layers, head_reduce):
    """[layers, heads, L, L] (CLS/EOS already stripped) → compact [C, L, L] float16 numpy.

      * restrict to the last `num_attn_layers` layers (None = all),
      * symmetrize ((A + A^T)/2) so the resulting DCCM prediction is symmetric,
      * head-reduce: 'mean' averages over heads (C = num_attn_layers) — memory-safe and gives
        the SAME C for every backbone, which is what makes the cross-method comparison fair;
        'none' keeps every head (C = num_attn_layers * heads) — much larger, short proteins only.
    """
    if num_attn_layers is not None and 0 < num_attn_layers < attn.shape[0]:
        attn = attn[-num_attn_layers:]
    attn = 0.5 * (attn + attn.transpose(-1, -2))

    if head_reduce == 'mean':
        attn = attn.mean(dim=1)                        # [Ls, L, L]
    elif head_reduce == 'none':
        Ls, H, L, _ = attn.shape
        attn = attn.reshape(Ls * H, L, L)              # [Ls*H, L, L]
    else:
        raise ValueError(f"head_reduce must be 'mean' or 'none', got {head_reduce}")
    return attn.to(torch.float16).cpu().numpy()


@torch.no_grad()
def _forward_esm_style(model, alphabet, sequence, device, repr_layer):
    """esm2 / dplm → (emb [L, D], attn [layers, heads, L, L])."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([('p', sequence)])
    tokens = tokens.to(device)
    if repr_layer is None:
        repr_layer = model.num_layers
    out = model(tokens, repr_layers=[repr_layer],
                need_head_weights=True, return_contacts=False)
    emb  = out['representations'][repr_layer][0, 1:-1, :]     # strip CLS/EOS
    attn = out['attentions'][0][:, :, 1:-1, 1:-1]             # [layers, heads, L, L]
    return emb, attn


@torch.no_grad()
def _forward_esmc(model, tokenizer, sequence, device):
    """esmc → (emb [L, D], attn [layers, heads, L, L]).

    Tokens are [<cls>, residues..., <eos>] as in ESM2, so the same 1:-1 strip applies.
    Requesting output_attentions makes each block compute attention explicitly instead of
    through SDPA (which never materialises the weight matrix), so the maps are exact.
    """
    enc = tokenizer([sequence], return_tensors='pt', add_special_tokens=True)
    ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    out = model(input_ids=ids, attention_mask=mask, output_attentions=True)
    emb = out.last_hidden_state[0, 1:-1, :]
    # out.attentions: tuple(layers) of [B, heads, T, T]
    attn = torch.stack(out.attentions, dim=0)[:, 0][:, :, 1:-1, 1:-1]
    return emb, attn


@torch.no_grad()
def _forward_prostt5(model, tokenizer, sequence, device):
    """prostt5 → (emb [L, D], attn [layers, heads, L, L]).

    Tokens are [<AA2fold>, residues..., </s>], so one prefix + one EOS are stripped — matching
    rmsf.py:get_residue_emb_prostt5.
    """
    import re as _re
    seq_clean = _re.sub(r'[UZOB]', 'X', sequence)
    seq_spaced = '<AA2fold> ' + ' '.join(list(seq_clean))
    ids = tokenizer([seq_spaced], add_special_tokens=True,
                    padding=False, return_tensors='pt').to(device)
    out = model(ids.input_ids, attention_mask=ids.attention_mask, output_attentions=True)
    emb = out.last_hidden_state[0, 1:-1, :]
    # out.attentions: tuple(layers) of [B, heads, T, T]
    attn = torch.stack(out.attentions, dim=0)[:, 0][:, :, 1:-1, 1:-1]
    return emb, attn


@torch.no_grad()
def _forward_seqdance(model, tokenizer, sequence, device):
    """seqdance → (emb [L, D], attn [layers, heads, L, L]).

    ESMwrap returns attention_map [B, T, T, layers*heads] (per-layer maps concatenated
    layer-major along the head axis, then permuted — model.py:108). We permute back to
    [layers*heads, T, T], strip CLS/EOS, then un-flatten into [layers, heads, L, L] using the
    underlying HF ESM config so the last-N-layer selection means the same thing as elsewhere.
    """
    raw_input = tokenizer([sequence], return_tensors='pt').to(device)
    out = model(raw_input, return_res_emb=True, return_attention_map=True,
                return_res_pred=False, return_pair_pred=False)
    emb = out['res_emb'][0, 1:-1, :]
    am = out['attention_map'][0].permute(2, 0, 1)[:, 1:-1, 1:-1]   # [layers*heads, L, L]

    C, L, _ = am.shape
    n_layers = n_heads = None
    cfg = getattr(getattr(model, 'esm2', None), 'config', None)
    if cfg is not None:
        n_layers = getattr(cfg, 'num_hidden_layers', None)
        n_heads  = getattr(cfg, 'num_attention_heads', None)
    if n_layers and n_heads and n_layers * n_heads == C:
        attn = am.reshape(n_layers, n_heads, L, L)
    else:
        # Unknown geometry — treat every channel as its own "layer" so the reduction still
        # works (last-N then selects the last N channels rather than the last N layers).
        print(f'  [warn] seqdance attention geometry unknown (C={C}); treating channels as layers')
        attn = am.reshape(C, 1, L, L)
    return emb, attn


def extract_emb_and_attn(method, model, extra, sequence, device, repr_layer=None,
                         num_attn_layers=4, head_reduce='mean'):
    """One forward pass → (emb [L, D] float32, attn [C, L, L] float16) for any backbone."""
    method = method.lower()
    if method in ('esm2', 'dplm'):
        emb, attn = _forward_esm_style(model, extra, sequence, device, repr_layer)
    elif method == 'esmc':
        emb, attn = _forward_esmc(model, extra, sequence, device)
    elif method == 'prostt5':
        emb, attn = _forward_prostt5(model, extra, sequence, device)
    elif method == 'seqdance':
        emb, attn = _forward_seqdance(model, extra, sequence, device)
    else:
        raise ValueError(f'No attention extractor for method {method}')
    emb = emb.float().cpu().numpy().astype(np.float32)
    return emb, _reduce_attn(attn, num_attn_layers, head_reduce)


def build_samples_attn(method, proteins, model, extra, analysis_path, device,
                       repr_layer=None, num_attn_layers=4, head_reduce='mean',
                       dccm_dir=None, replicate=None):
    """Build [{pid, emb, attn, dccm}] for the given proteins + one attention-capable model.

    Proteins whose ground-truth DCCM is missing or shape-mismatched are skipped, mirroring
    build_samples in data_dccm.py (but here embeddings + attention come from one forward pass).
    """
    samples = []
    for prot in proteins:
        pid, seq = prot['pid'], prot['sequence']
        try:
            emb, attn = extract_emb_and_attn(
                method, model, extra, seq, device, repr_layer=repr_layer,
                num_attn_layers=num_attn_layers, head_reduce=head_reduce)
        except Exception as e:
            print(f'  [skip] {pid}: embedding/attention extraction failed: {e}')
            continue

        L = len(seq)
        if emb.shape[0] != L or attn.shape[-1] != L:
            print(f'  [skip] {pid}: emb len {emb.shape[0]} / attn len {attn.shape[-1]} '
                  f'≠ seq len {L}')
            continue

        dccm = load_protein_dccm(pid, analysis_path,
                                 dccm_dir=dccm_dir, replicate=replicate)
        if dccm is None:
            print(f'  [skip] {pid}: ground-truth DCCM unavailable')
            continue
        if dccm.shape != (L, L):
            print(f'  [skip] {pid}: DCCM shape {dccm.shape} != emb len {L}')
            continue

        samples.append({
            'pid':  pid,
            'emb':  emb.astype(np.float32),
            'attn': attn,                        # float16 [C, L, L]
            'dccm': dccm.astype(np.float32),
        })

    print(f'  built {len(samples)} samples (from {len(proteins)} proteins)')
    return samples
