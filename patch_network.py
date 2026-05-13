import sys

network_file = "/home/daksh/neas_ablations/NeAS-2/src/network/network.py"
with open(network_file, "r") as f:
    network_code = f.read()

# Add IndependentAttenuationMLP
independent_mlp_code = """
class IndependentAttenuationMLP(nn.Module):
    \"\"\"Independent attenuation networks for K materials without shared backbone.\"\"\"
    def __init__(self, input_dim, material_activations, encoding_type='hash'):
        super().__init__()
        self.heads = nn.ModuleList()
        # Original NeAS used distinct MLPs with 64 hidden units for hash, 256 for freq.
        hidden_dim = 64 if encoding_type == 'hash' else 256
        num_layers = 2 if encoding_type == 'hash' else 4
        
        for alpha, beta in material_activations:
            mlp = MLPBlock(input_dim, hidden_dim, 1, num_layers)
            activation = CustomActivation(alpha, beta)
            self.heads.append(nn.Sequential(mlp, activation))
            
    def forward(self, features):
        return [head(features) for head in self.heads]
        
"""

old_class = "class SDFMLPWrapperKM"
network_code = network_code.replace(old_class, independent_mlp_code + old_class)

# Add hard selector
selector_code = """
def hard_material_selector(distances, raw_attenuations, boundary_values):
    \"\"\"Selector function for choosing between two attenuation coefficients using original hard selector.
    
    distances: K=2
    raw_attenuations: K=2 [B, 1]
    boundary_values: K=2 [B]
    \"\"\"
    d2 = distances[1]
    # boundary * raw_att. squeeze to [B]
    mu1 = boundary_values[0] * raw_attenuations[0].squeeze(-1)
    mu2 = boundary_values[1] * raw_attenuations[1].squeeze(-1)
    
    # original logic: torch.where(d2 < 0, mu2, mu1)
    return torch.where(d2 < 0, mu2, mu1)
    
"""

old_class2 = "def nested_material_selector"
network_code = network_code.replace(old_class2, selector_code + old_class2)

with open(network_file, "w") as f:
    f.write(network_code)
