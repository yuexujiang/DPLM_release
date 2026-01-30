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




class IDREncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.esm2, self.alphabet = prepare_adapter_h_model(configs)

        # self.esm2, self.alphabet = prepare_esm_model(configs, logging)
        self.batch_converter = self.alphabet.get_batch_converter()
        # Linear head for residue-level classification (2 classes: order/disorder)
        self.head = nn.Linear(self.esm2.embed_dim, 1) 
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, seqs):
        # Prepare tokens
        batch_data = [("seq_" + str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = self.batch_converter(batch_data)

        if torch.cuda.is_available():
            tokens=tokens.to(self.device)

        # Get residue-level embeddings [Batch, SeqLen, EmbedDim]
        results = self.esm2(tokens, repr_layers=[self.esm2.num_layers])
        residue_features = results['representations'][self.esm2.num_layers]
        
        # Project to scores [Batch, SeqLen, 1]
        # Remove CLS/EOS tokens to match raw sequence length
        logits = self.head(residue_features[:, 1:-1, :]) 
        # logits = self.sigmoid(logits)
        return torch.sigmoid(logits).squeeze(-1)


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
    encoder = IDREncoder(configs=configs)
 
    return encoder

def load_model_idr(config_path, model_location):
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = Box(config_file)
    model = prepare_models(configs)
    model = load_checkpoints_infer(model, model_location)
    model.eval()
    return model