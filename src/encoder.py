import torch
import torch.nn as nn
import torch.nn.functional as F

from manifold import *
from utils_layer import *

class TransformerBlock(nn.Module):
    def __init__(self, in_features, out_features, dim_feedforward, nhead, num_layers):
        super().__init__()
        
        tf_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead,
                                              dim_feedforward=dim_feedforward, batch_first=True
                                             )
        self.TF_Block = nn.TransformerEncoder(tf_layer, num_layers=num_layers)
        self.Out = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        z = self.TF_Block(x)
        z = self.Out(z)
        
        return z

class EncodingLayer(nn.Module):
    def __init__(self, dim_data, dim_embedding, layer_configs):
        super().__init__()

        # Layers -- for test
        T, D = dim_data
        dim_initial = 2 ** (torch.log2(torch.Tensor([T*D])).ceil().int().item())
        in_features = (dim_initial / 4 ** torch.arange(3)).int().tolist()
        out_features = in_features[1:] + [dim_embedding]
        
        self.Span = nn.Linear(D, dim_initial)
        self.LN = nn.LayerNorm([T, dim_initial])
        self.TF_Block1 = TransformerBlock(in_features=in_features[0], out_features=out_features[0], 
                                          dim_feedforward=layer_configs.get('dim_feedforward_1', in_features[0]*2), 
                                          nhead=layer_configs.get('nhead_1', 4), 
                                          num_layers=layer_configs.get('num_layers_1', 2)
                                         )
        self.TF_Block2 = TransformerBlock(in_features=in_features[1], out_features=out_features[1], 
                                          dim_feedforward=layer_configs.get('dim_feedforward_2', in_features[1]*2), 
                                          nhead=layer_configs.get('nhead_2', 4), 
                                          num_layers=layer_configs.get('num_layers_2', 2)
                                         )
        self.TF_Block3 = TransformerBlock(in_features=in_features[2], out_features=out_features[2], 
                                          dim_feedforward=layer_configs.get('dim_feedforward_3', in_features[2]*2), 
                                          nhead=layer_configs.get('nhead_3', 4), 
                                          num_layers=layer_configs.get('num_layers_3', 2)
                                         )
                                         
    def forward(self, x):
        local_emb = self.Span(x)
        local_emb = self.LN(local_emb)
        
        local_emb = self.TF_Block1(local_emb)
        local_emb = self.TF_Block2(local_emb)
        local_emb = self.TF_Block3(local_emb)
        
        local_emb = local_emb.sum(dim=1)
        
        return local_emb

class Encoder(nn.Module):
    def __init__(self, dim_data, dim_embedding, target_manifold, 
                 layer_configs={}, device='cpu'):
        super().__init__()
        
        # Encoding Layer
        self.encoding_layer = EncodingLayer(dim_data, dim_embedding, layer_configs)
        
        # Target Manifold
        self.manifold = self._set_target_manifold(target_manifold, dim_embedding)
        
        # Misc.
        self.dim_data = dim_data
        self.dim_embedding = dim_embedding
        self.layer_configs = layer_configs
        
        self.device = device
        self.to(device)
        
    def forward(self, x):
        local_emb = self.encoding_layer(x.to(self.device))
        
        return local_emb
        
    def _set_target_manifold(self, target_manifold, dim_embedding):
        manifolds = {'euclidean':EuclideanSpace, 
                     'sphere':Hypersphere, 
                     'sphere_sp':HypersphereStereographic, 
                     'p_plane':PoincareHalfplane
                    }
        
        if target_manifold in manifolds.keys():
            return manifolds[target_manifold](dim_embedding)
        else:
            raise ValueError(f'Unknown value : "target_manifold={target_manifold}"\n    - Consider one of ["euclidean", "sphere", "sphere_sp", "p_plane"].')
        
    def distance(self, x1, x2):
        local_coords1 = self.forward(x1)
        local_coords2 = self.forward(x2)
        
        return self.manifold.distance(local_coords1, local_coords2)
    
    def pullback_metric(self, x):
        N, T, D = x.shape
        
        if T*D > self.dim_embedding:
            jacobian_mod = 'rev'
        else:
            jacobian_mod = 'fwd'
        
        local_coords = self(x.to(self.device))
        
        J = torch.zeros(N, self.dim_embedding, T, D)
        for i in range(N):
            J[i] = jacobian(self, x[[i]], mode=jacobian_mod)
        J = torch.flatten(J, start_dim=-2)
        H = self.manifold.metric(local_coords)
        
        P_metric = J.transpose(1, 2) @ H @ J
        
        return P_metric
        
        
    def get_points_on_manifold(self, x):
        local_coords = self.manifold(x.to(self.device))
        points = self.manifold.local_to_manifold(local_coords)
        
        return points