"""
model_ddg_v2.py — Mutation-site-aware ddG regressor (EncoderSiteAware).

Motivation
----------
The original Encoder (model_ddg.py) predicts ΔΔG as
    linear_head(mean_pool(MT)) − linear_head(mean_pool(WT))
over the *entire* token sequence. Because the head is linear this collapses to a
whole-protein mean difference, so the single-residue mutation signal is diluted by
~1/L, and AdaptiveAvgPool1d even averages in the BOS/EOS/padding tokens.

EncoderSiteAware instead builds features focused on the mutated residue(s):
    site_from / site_to : mean of the ESM2 representations at the mutated positions
    global_from / global_to : masked mean pooling over real residues (no BOS/EOS/pad)
    feature = concat[ site_to − site_from, global_to − global_from, site_to, site_from ]
and feeds them through a small MLP head. The model is *direction-aware*
(from_seq → to_seq), not hard-coded antisymmetric, so inverse-mutation augmentation
(data_ddg_v2.py) contributes real gradient signal.

The ESM2 + Houlsby-adapter backbone is built by reusing prepare_adapter_h_model /
prepare_esm_model from model_ddg.py — identical adapter_0-frozen / adapter_1-trainable
setup and the same DPLM-checkpoint compatibility (load_esm2_checkpoint).
"""

import torch
from torch import nn

# Reuse backbone construction + param logging from the original model file.
import numpy as np
import esm
import esm_adapterH
from peft import LoraConfig, get_peft_model
# (prepare_adapter_h_model / prepare_esm_model / print_trainable_parameters inlined below)


# ── inlined from model_ddg.py (the pre-v2 module, removed
#    in the public release; only these helpers were ever used from it) ─────
def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )

def verify_data_types(model, logging=None):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        if logging:
           logging.info(f"{k}, {v}, {v / total}")

def prepare_esm_model(configs, logging=None):
    if logging:
        logging.info("use ESM model")
    
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm.pretrained, model_name, None)
    model, alphabet = model_constructor()
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

        # only freeze all the parameters once at the beginning. then open some layers later

    if configs.encoder.lora.enable:
        if logging:
           logging.info('enable LoRa on top of esm model')
        #target_modules = [
        #    "k_proj", "v_proj", "q_proj","fc1", "fc2"]
        if hasattr(configs.encoder.lora,"lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
        
    
    if configs.encoder.tune_embedding:
        if logging:
           logging.info('make esm embedding parameters trainable')
        
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet

def prepare_adapter_h_model(configs, logging=None):
    if logging:
       logging.info("use adapterH ESM model")
    
    adapter_args = configs.encoder.adapter_h
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm_adapterH.pretrained, model_name, None)
    model, alphabet = model_constructor(adapter_args)
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    if configs.encoder.adapter_h.enable:
      if not isinstance(configs.encoder.adapter_h.freeze_adapter_layers, list):
        configs.encoder.adapter_h.freeze_adapter_layers = [configs.encoder.adapter_h.freeze_adapter_layers]
    
    if configs.encoder.fine_tune.enable:
      if not isinstance(configs.encoder.fine_tune.freeze_adapter_layers, list):
        configs.encoder.fine_tune.freeze_adapter_layers = [configs.encoder.fine_tune.freeze_adapter_layers]
    
    if configs.encoder.lora.enable:
        if logging:
           logging.info('enable LoRa on top of adapterH model')
        if hasattr(configs.encoder.lora,"lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
            #modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    
    
    # only freeze all the parameters once at the beginning. then open some layers later
    #only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
      for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
        if not value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    param.requires_grad = True
    
    # only freeze all the parameters once at the beginning. then open some layers later,but because
    # of fine_tune, adapter layers might be tunable.
    #change on 1/15/2024 not need to use freeze_adapter_layers to control fine-tune part! use another parameter instead and must after setting of freeze_adapter_layers
    if configs.encoder.fine_tune.enable: #only see fine_tune.freeze_adapter_layers when fine-tune is available
       for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
        if value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    print("freeze adapter in fine-tune")
                    param.requires_grad = False
    #"""
    
    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet


class EncoderSiteAware(nn.Module):
    def __init__(self, logging, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs, logging)
        else:
            self.esm2, self.alphabet = prepare_esm_model(configs, logging)

        self.batch_converter = self.alphabet.get_batch_converter()
        embed_dim = self.esm2.embed_dim

        # Feature = [site_diff, global_diff, site_to, site_from] → 4 * embed_dim
        feat_dim   = 4 * embed_dim
        hidden_dim = int(getattr(configs.encoder, 'head_hidden_dim', 512))
        dropout    = float(getattr(configs.encoder, 'head_dropout', 0.2))
        num_classes = configs.encoder.num_classes

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.device = configs.train_settings.device
        self.configs = configs

    # ── helpers ──────────────────────────────────────────────────────────────

    def _encode(self, seqs):
        """Run ESM2 on a list of sequences → (reps [B, T, D], token_ids [B, T])."""
        batch = [("seq_" + str(i), str(seqs[i])) for i in range(len(seqs))]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        # NOTE: no torch.no_grad() — adapter_1 is trained via grads through esm2.
        reps = self.esm2(tokens, repr_layers=[self.esm2.num_layers])['representations'][self.esm2.num_layers]
        return reps, tokens

    def _residue_mask(self, tokens):
        """Boolean mask [B, T] of real residues (exclude BOS/EOS/pad)."""
        mask = (
            (tokens != self.alphabet.padding_idx)
            & (tokens != self.alphabet.cls_idx)
            & (tokens != self.alphabet.eos_idx)
        )
        return mask

    @staticmethod
    def _masked_mean(reps, mask):
        """Masked mean over the token dimension. reps [B, T, D], mask [B, T]."""
        m = mask.unsqueeze(-1).to(reps.dtype)          # [B, T, 1]
        denom = m.sum(dim=1).clamp(min=1.0)            # [B, 1]
        return (reps * m).sum(dim=1) / denom           # [B, D]

    def _site_mean(self, reps, mask, mut_pos):
        """Mean of representations at the mutated token positions (index = residue+1
        for the BOS offset). Falls back to the masked global mean when a sample has
        no recorded mutation positions."""
        B, T, D = reps.shape
        out = reps.new_zeros(B, D)
        global_mean = self._masked_mean(reps, mask)
        for i in range(B):
            pos = mut_pos[i]
            if not pos:
                out[i] = global_mean[i]
                continue
            # +1 for the prepended BOS token; guard against truncation past T-1.
            tok_idx = [p + 1 for p in pos if (p + 1) < T]
            if not tok_idx:
                out[i] = global_mean[i]
                continue
            idx = torch.tensor(tok_idx, device=reps.device, dtype=torch.long)
            out[i] = reps[i].index_select(0, idx).mean(dim=0)
        return out

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, from_seqs, to_seqs, mut_pos):
        reps_from, tok_from = self._encode(from_seqs)
        reps_to,   tok_to   = self._encode(to_seqs)

        mask_from = self._residue_mask(tok_from)
        mask_to   = self._residue_mask(tok_to)

        site_from   = self._site_mean(reps_from, mask_from, mut_pos)
        site_to     = self._site_mean(reps_to,   mask_to,   mut_pos)
        global_from = self._masked_mean(reps_from, mask_from)
        global_to   = self._masked_mean(reps_to,   mask_to)

        feature = torch.cat(
            [site_to - site_from, global_to - global_from, site_to, site_from],
            dim=-1,
        )
        return self.head(feature).squeeze(-1)   # [B]


def prepare_models_v2(configs, logging):
    """Factory mirroring model_ddg.prepare_models."""
    encoder = EncoderSiteAware(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder (site-aware) parameters: ' + str(sum(p.numel() for p in encoder.parameters())))
    return encoder
