import torch

def random_perturbation(data, noise_scale):
    diff = torch.diff(data, n=1, dim=1)
    diff_mean = diff.mean(dim=1, keepdims=True)
    diff_std = diff.std(dim=1, keepdims=True)
    
    noise = torch.randn(data.shape)
    
    perturbed_data = data + noise_scale * (diff_std * noise + diff_mean)
    
    return perturbed_data
    
def random_rescaling(data, scale_band):
    scale_min, scale_max = scale_band
    
    scales = torch.rand(data.shape) * (scale_max - scale_min) + scale_min
    rescaled_data = data * scales
    
    return rescaled_data
    
def mining_Frobenius_norm(data, near, far):
    N, T, D = data.shape
    data_flattened = data.view(N, -1, 1)
    diff = data_flattened.permute(1, 0, 2) - data_flattened.permute(1, 2, 0)
    
    F_norm_sq = diff.square().sum(dim=0)

    idx_near = int(N * near)
    idx_far = int(N * far)

    idx_mined = F_norm_sq.sort().indices[:, [idx_near, idx_far]]
    
    data_near = data[idx_mined[:, 0]]
    data_far = data[idx_mined[:, 1]]
    
    return data_near, data_far
    
class DataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self._generate_new_indices()
        
    def _generate_new_indices(self, ):
        N = self.data.shape[0]
        indices = torch.randperm(N)
        self.batch_indices = list(torch.split(indices, self.batch_size))
        
    def get_batch(self, as_triplet=False, 
                  noise_scale=.5, scale_band=(0.8, 1.2), 
                  near=.25, far=.75
                 ):
        batch_index = self.batch_indices.pop(0)
        batch = self.data[batch_index]
        
        if as_triplet:
            # Mining
            batch_pos1 = random_perturbation(batch, noise_scale)
            batch_neg1 = random_rescaling(batch, scale_band)
            batch_pos2, batch_neg2 = mining_Frobenius_norm(batch, near, far)
            
            # Concat
            batch = (batch, batch_pos1, batch_pos2, batch_neg1, batch_neg2)
            
        if self.batch_indices.__len__() == 0:
            self._generate_new_indices()
            
        return batch
    