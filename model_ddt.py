import torch
from torch import nn
import esm_adapterH
from utils.utils import *



def prepare_adapter_h_model(configs):
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
    
    
    # only freeze all the parameters once at the beginning. then open some layers later
    #only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
      for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
        if not value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    param.requires_grad = True
    
    return model, alphabet

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.esm2, self.alphabet = prepare_adapter_h_model(configs)

        self.batch_converter = self.alphabet.get_batch_converter()
        embed_dim = self.esm2.embed_dim
        
        self.head = nn.Linear(embed_dim, configs.encoder.num_classes)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, wt_seq, mt_seq):


        batch_wt_seq = [("seq_" + str(i), str(wt_seq[i])) for i in range(len(wt_seq))]
        batch_labels, batch_strs, batch_tokens_wt = self.batch_converter(batch_wt_seq)

        batch_mt_seq = [("seq_" + str(i), str(mt_seq[i])) for i in range(len(mt_seq))]
        batch_labels, batch_strs, batch_tokens_mt = self.batch_converter(batch_mt_seq)

        batch_tokens_wt = batch_tokens_wt.to(self.device)
        batch_tokens_mt = batch_tokens_mt.to(self.device)

        features_wt = self.esm2(batch_tokens_wt,
                             repr_layers=[self.esm2.num_layers])['representations'][self.esm2.num_layers]
    
        features_mt = self.esm2(batch_tokens_mt,
                             repr_layers=[self.esm2.num_layers])['representations'][self.esm2.num_layers]

        transposed_feature_wt = features_wt.transpose(1, 2)
        transposed_feature_mt = features_mt.transpose(1, 2)
        pooled_features_wt = self.pooling_layer(transposed_feature_wt).squeeze(2)
        pooled_features_mt = self.pooling_layer(transposed_feature_mt).squeeze(2)
        wt_score = self.head(pooled_features_wt)
        mt_score = self.head(pooled_features_mt)
        return (mt_score - wt_score).squeeze(-1)


def prepare_models(configs):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = Encoder(configs=configs)


    return encoder


def load_model_ddt(config_path, model_location):
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = Box(config_file)
    model = prepare_models(configs)
    model = load_checkpoints_infer(model, model_location)
    model.eval()
    return model