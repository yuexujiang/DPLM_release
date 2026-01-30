from box import Box
import torch
import esm_adapterH
import yaml
from collections import OrderedDict



def load_checkpoints(model,checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #this model does not contain esm2
    new_ordered_dict = OrderedDict()
    for key, value in checkpoint['state_dict1'].items():
            key = key.replace("esm2.","")
            new_ordered_dict[key] = value
    model.load_state_dict(new_ordered_dict, strict=False)
    print("checkpoints were loaded from " + checkpoint_path)


def extract_emb_perseq(model, alphabet, sequence):
    batch_converter = alphabet.get_batch_converter()
    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    if torch.cuda.is_available():
         batch_tokens=batch_tokens.cuda()
    with torch.no_grad():
        wt_representation = model(batch_tokens,repr_layers=[model.num_layers])["representations"][model.num_layers]
    wt_representation = wt_representation.squeeze(0) #only one sequence a time
    return wt_representation

def load_model(config_path, model_location):
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = Box(config_file)
    model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
    load_checkpoints(model, model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    return model,alphabet

########################################################
def load_checkpoints_infer(model, model_location):
    # If the 'resume' flag is True, load the saved model checkpoints.
    model_checkpoint = torch.load(model_location, map_location='cpu')
    #net.load_state_dict(model_checkpoint['state_dict1'], strict=False)
    if 'state_dict1' in model_checkpoint:
        #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
        model.load_state_dict(model_checkpoint['state_dict1'], strict=False)
    
    # Return the loaded model and the epoch to start training from.
    return model

