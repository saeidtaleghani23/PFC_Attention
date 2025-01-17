from pfc_attention import pfc_classifier
import torch
import yaml

# Read the config file
yaml_file_path = "/media/saeid/Crucial/PFC_Attention/MDPI_Github/config/pfc_config.yaml"
# Open and read the YAML file
with open(yaml_file_path, "r") as file:
    pfc_config = yaml.load(file, Loader=yaml.FullLoader)

# create a model with the default values
pfc_model = pfc_classifier(pfc_config)
pfc_model.eval()
# create a simple sample
Input = torch.randn(1, pfc_config['MODEL']['in_channels'], pfc_config['MODEL']['img_size'], pfc_config['MODEL']['img_size'])
# Output
Output = pfc_model(Input)
# Output shape must be 1x5
# Assert that the output shape is 1x5
assert Output.shape == (1, pfc_config['MODEL']['num_classes']), f"Output shape is {Output.shape}, expected (1, {pfc_config['MODEL']['num_classes']})"
    
print(f"Output shape is correct:{Output.shape}\nModel works!")
