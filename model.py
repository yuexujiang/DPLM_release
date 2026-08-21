import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import esm
import esm_adapterH
from peft import LoraConfig, get_peft_model
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode, minibatch_size):
    N = minibatch_size
    if mode == "struct_struct":
        a11 = logits[0:N, 1:N]
        return a11.mean().item()

    if mode == "struct_seq":
        a12 = logits[0:N, N:2 * N - 1]
        return a12.mean().item()

    if mode == "seq_struct":
        a21 = logits[N:2 * N, 1:N]
        return a21.mean().item()
    if mode == "seq_seq":
        a22 = logits[N:2 * N, N:2 * N - 1]
        return a22.mean().item()


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x


class ESM2(nn.Module):
    def __init__(self, esm2_pretrain, logging,
                 accelerator,
                 configs,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        super(ESM2, self).__init__()
        if configs.model.esm_encoder.adapter_h.enable:
            if accelerator.is_main_process:
                logging.info("use adapter H")
            adapter_args = configs.model.esm_encoder.adapter_h
            esm2_constructors = {
                         "esm2_t36_3B_UR50D": esm_adapterH.pretrained.esm2_t36_3B_UR50D,
                         "esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D,
                         "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D,
                         "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D,
                         "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D,
                         }
            # Only the selected model is constructed (lazy dispatch) — the dict above
            # previously held already-called constructors as values, which meant Python
            # built ALL FIVE model sizes (including the 3B one) on every run regardless
            # of which esm2_pretrain was actually selected.
            self.esm2, self.alphabet = esm2_constructors[esm2_pretrain](adapter_args)
        else:
            esm2_constructors = {
                         "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
                         "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
                         }
            self.esm2, self.alphabet = esm2_constructors[esm2_pretrain]()

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():
            p.requires_grad = False

        if configs.model.esm_encoder.adapter_h.enable:
            if not isinstance(configs.model.esm_encoder.adapter_h.freeze_adapter_layers, list):
                configs.model.esm_encoder.adapter_h.freeze_adapter_layers = [configs.model.esm_encoder.adapter_h.freeze_adapter_layers]
            for adapter_idx, value in enumerate(configs.model.esm_encoder.adapter_h.freeze_adapter_layers):
                if not value:
                    for name, param in self.esm2.named_parameters():
                        adapter_name = f"adapter_{adapter_idx}"
                        if adapter_name in name:
                            param.requires_grad = True

        if configs.model.esm_encoder.lora.enable:
            if accelerator.is_main_process:
                logging.info('use lora for esm v2')
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
            target_modules = []
            if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                start_layer_idx = np.max([self.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                for idx in range(start_layer_idx, self.num_layers):
                    for layer_name in lora_targets:
                        target_modules.append(f"layers.{idx}.{layer_name}")

            peft_config = LoraConfig(
                inference_mode=False,
                r=configs.model.esm_encoder.lora.r,
                lora_alpha=configs.model.esm_encoder.lora.alpha,
                target_modules=target_modules,
                lora_dropout=configs.model.esm_encoder.lora.dropout,
                bias="none",
            )
            self.peft_model = get_peft_model(self.esm2, peft_config)
        elif configs.model.esm_encoder.fine_tuning.enable:
            if accelerator.is_main_process:
                logging.info('fine-tune esm v2')
            unfix_last_layer = configs.model.esm_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1
                    continue
                for p in layer.parameters():
                    p.requires_grad = True

            if unfix_last_layer != 0:
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True

        self.projectors_residue = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=residue_inner_dim,
                                          num_layers=residue_num_projector,
                                          out_dim=residue_out_dim)

        self.projectors_protein = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=protein_inner_dim,
                                          num_layers=protein_num_projector,
                                          out_dim=protein_out_dim)

    def forward(self, x, return_logits=False, return_embedding=False):
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)
        if return_logits:
            prediction_scores = outputs["logits"]
            return prediction_scores
        else:
            residue_feature = outputs['representations'][self.num_layers]
            mask = (x != self.alphabet.padding_idx)
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature_embedding = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom
            graph_feature = self.projectors_protein(graph_feature_embedding)
            mask = ((x != self.alphabet.padding_idx) & (x != self.alphabet.cls_idx) & (
                    x != self.alphabet.eos_idx))
            residue_feature_embedding = residue_feature[mask]
            residue_feature = self.projectors_residue(residue_feature_embedding)
            if return_embedding:
                return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
            else:
                return graph_feature, residue_feature


class VIVIT(nn.Module):
    def __init__(self, vivit_pretrain, logging,
                 accelerator,
                 configs,
                 dim_mlp=768,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        super(VIVIT, self).__init__()

        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        if hasattr(configs.model.MD_encoder, "fine_tuning_projct") and not configs.model.MD_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False

    def forward(self, x, return_logits=False, return_embedding=False):
        if return_logits:
            print("print something")
        else:
            graph_feature = self.projectors_protein(x)

        if return_embedding:
            return graph_feature, x
        else:
            return graph_feature


class MaskedLMDataCollator:
    """Data collator for masked language modeling."""

    def __init__(self, batch_converter, mlm_probability=0.15):
        self.mlm_probability = mlm_probability
        self.special_token_indices = [batch_converter.alphabet.cls_idx,
                                batch_converter.alphabet.padding_idx,
                                batch_converter.alphabet.eos_idx,
                                batch_converter.alphabet.unk_idx,
                                batch_converter.alphabet.mask_idx]
        self.vocab_size = batch_converter.alphabet.all_toks.__len__()
        self.mask_idx = batch_converter.alphabet.mask_idx

    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]

    def mask_tokens(self, batch_tokens):
        inputs = batch_tokens.clone().to(batch_tokens.device)
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size,
                                     labels.shape, dtype=torch.long).to(batch_tokens.device)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


class SimCLR(nn.Module):
    def __init__(self, model_seq, model_x, configs):
        super(SimCLR, self).__init__()
        self.model_seq = model_seq
        self.model_x = model_x
        self.temperature = configs.train_settings.temperature
        self.n_views = configs.train_settings.n_views
        self.configs = configs
        # Learnable temperature (CLIP-style): logit_scale = log(1/τ), so τ = 1/exp(logit_scale)
        # Created only when learnable_temperature: True in config; absent otherwise.
        if getattr(configs.train_settings, 'learnable_temperature', False):
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1.0 / configs.train_settings.temperature)
            )

    def forward(self, graph=None, batch_tokens=None, mode=False, return_embedding=False, return_logits=False):
        if mode == 'sequence':
            return self.model_seq(batch_tokens, return_embedding=return_embedding, return_logits=return_logits)
        elif mode == 'structure':
            return self.model_x(graph, return_embedding=return_embedding)
        elif mode == "MD" or mode == 'vivit' or mode == 'MD_tune':
            return self.model_x(graph)
        else:
            if self.configs.model.X_module == "MD" or self.configs.model.X_module == "vivit":
                features_MD = self.model_x(graph)
                if return_embedding:
                    features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding = self.model_seq(batch_tokens, return_embedding=return_embedding)
                    return features_MD, features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding
                else:
                    features_seq, residue_seq = self.model_seq(batch_tokens)
                    return features_MD, features_seq, residue_seq

    def forward_sequence(self, batch_tokens):
        features_seq, residue_seq = self.model_seq(batch_tokens)
        return features_seq, residue_seq

    def forward_x(self, graph):
        if self.configs.model.X_module == "MD":
            features_MD = self.model_x(graph)
            return features_MD


def clip_infonce(features_struct, features_seq, temperature, accelerator):
    B = features_struct.shape[0]

    z_s = F.normalize(features_struct, dim=1)
    z_q = F.normalize(features_seq, dim=1)

    logits = (z_s @ z_q.T) / temperature
    targets = torch.arange(B, device=accelerator.device)

    return logits, targets


def info_nce_loss(features_struct, features_seq, n_views, temperature, accelerator):
    batch_size = len(features_struct)

    labels = torch.cat([torch.arange(batch_size, device=accelerator.device) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(accelerator.device)

    features_struct = F.normalize(features_struct, dim=1)
    features_seq = F.normalize(features_seq, dim=1)
    features = torch.cat([features_struct, features_seq], dim=0)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=accelerator.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=accelerator.device)

    logits = logits / temperature
    return logits, labels


def clip_infonce_multipos(features_seq, features_md_list, temperature, accelerator):
    """Multi-positive InfoNCE (SupCon-style, Khosla et al. 2020).

    For each sequence i, *all* R MD replicate embeddings at position i are
    treated as positives (capturing the same protein from different MD runs).

    Args:
        features_seq      : [B, D]  — raw (un-normalised) sequence embeddings.
        features_md_list  : list of R tensors, each [B, D] — one per replicate.
                            features_md_list[r][i] is replicate r of protein i.
        temperature       : scalar τ (or tensor from logit_scale).
        accelerator       : Accelerator (for device access).

    Returns:
        loss     : scalar InfoNCE loss.
        logits   : [B*R, B] logits from the MD→seq direction (for logging).
        targets  : [B*R] target indices (for top-1 accuracy logging).
    """
    B = features_seq.shape[0]
    R = len(features_md_list)

    z_seq = F.normalize(features_seq, dim=1)               # [B,   D]
    # Stack replicates: md_cat[r*B + i] = replicate r of protein i
    md_cat = torch.cat(features_md_list, dim=0)            # [B*R, D]
    z_md   = F.normalize(md_cat, dim=1)

    # ── Seq → MD  [B, B*R] ──────────────────────────────────────────────────
    # For seq_i the positives are at columns {r*B + i | r = 0..R-1}.
    logits_s2m = (z_seq @ z_md.T) / temperature            # [B, B*R]
    mask = torch.zeros(B, B * R, dtype=torch.bool, device=z_seq.device)
    for r in range(R):
        mask[torch.arange(B), r * B + torch.arange(B)] = True
    # SupCon: −(1/R) · Σ_positives log_softmax(logits)[positive]
    log_sm   = F.log_softmax(logits_s2m, dim=1)            # [B, B*R]
    loss_s2m = -log_sm[mask].view(B, R).mean()

    # ── MD → Seq  [B*R, B] ──────────────────────────────────────────────────
    # Replicate r of protein i (row r*B+i) should match sequence i.
    logits_m2s = (z_md @ z_seq.T) / temperature            # [B*R, B]
    targets    = torch.arange(B, device=z_seq.device).repeat(R)  # [0..B-1, repeated R times]
    loss_m2s   = F.cross_entropy(logits_m2s, targets)

    loss = 0.5 * (loss_s2m + loss_m2s)
    return loss, logits_m2s, targets


def print_trainable_parameters(model, logging):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models(logging, configs, accelerator):
    model_seq = ESM2(configs.model.esm_encoder.model_name,
                     accelerator=accelerator,
                     residue_inner_dim=configs.model.esm_encoder.residue_inner_dim,
                     protein_inner_dim=configs.model.esm_encoder.protein_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging)

    model_MD = VIVIT(configs.model.MD_encoder.model_name,
                 accelerator=accelerator,
                 residue_inner_dim=configs.model.MD_encoder.residue_inner_dim,
                 protein_inner_dim=configs.model.MD_encoder.protein_inner_dim,
                 residue_out_dim=configs.model.residue_out_dim,
                 protein_out_dim=configs.model.protein_out_dim,
                 residue_num_projector=configs.model.residue_num_projector,
                 protein_num_projector=configs.model.protein_num_projector,
                 configs=configs, logging=logging,
                 dim_mlp=configs.model.MD_encoder.dim_mlp)

    if accelerator.is_main_process:
        print_trainable_parameters(model_seq, logging)
        print_trainable_parameters(model_MD, logging)

    simclr = SimCLR(model_seq, model_MD, configs=configs)
    return simclr


if __name__ == '__main__':
    print('test')
