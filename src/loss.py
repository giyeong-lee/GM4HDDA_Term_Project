import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, distance):
        super().__init__()
        self.distance = distance
        
    def _sanity_check(self, dist_pos, dist_neg):
        n_na_dist_pos = dist_pos.isnan().sum()
        n_np_dist_pos = (dist_pos <= 0).sum()
        n_na_dist_neg = dist_neg.isnan().sum()
        n_np_dist_neg = (dist_neg <= 0).sum()
        
        assert n_na_dist_pos == 0, f'n_na_dist_pos = {n_na_dist_pos}\n {dist_pos}'
        assert n_np_dist_pos == 0, f'n_np_dist_pos = {n_np_dist_pos}\n {dist_pos}'
        assert n_na_dist_neg == 0, f'n_na_dist_neg = {n_na_dist_neg}\n {dist_neg}'
        assert n_np_dist_neg == 0, f'n_np_dist_neg = {n_np_dist_neg}\n {dist_neg}'
        
        
    def forward(self, local_emb, eps=1e-9):
        self.local_emb = local_emb
        local_emb_ref, local_emb_pos1, local_emb_pos2, local_emb_neg1, local_emb_neg2 = torch.tensor_split(local_emb, 5, dim=0)
        
        local_emb_ref = local_emb_ref.repeat(2, 1)
        local_emb_pos = torch.cat((local_emb_pos1, local_emb_pos2), dim=0)
        local_emb_neg = torch.cat((local_emb_neg1, local_emb_neg2), dim=0)
        
        dist_pos = self.distance(local_emb_ref, local_emb_pos) + eps
        dist_neg = self.distance(local_emb_ref, local_emb_neg) + eps
        
        self._sanity_check(dist_pos, dist_pos)
        
        _loss = (dist_pos.log() - dist_neg.log()).mean()
        
        return _loss
        
class TripletLoss(nn.Module):
    def __init__(self, distance, margin=0.1):
        super().__init__()
        self.distance = distance
        self.margin = margin
        
    def _sanity_check(self, dist_pos, dist_neg):
        n_na_dist_pos = dist_pos.isnan().sum()
        n_np_dist_pos = (dist_pos <= 0).sum()
        n_na_dist_neg = dist_neg.isnan().sum()
        n_np_dist_neg = (dist_neg <= 0).sum()
        
        assert n_na_dist_pos == 0, f'n_na_dist_pos = {n_na_dist_pos}\n {dist_pos}'
        assert n_np_dist_pos == 0, f'n_np_dist_pos = {n_np_dist_pos}\n {dist_pos}'
        assert n_na_dist_neg == 0, f'n_na_dist_neg = {n_na_dist_neg}\n {dist_neg}'
        assert n_np_dist_neg == 0, f'n_np_dist_neg = {n_np_dist_neg}\n {dist_neg}'
        
        
    def forward(self, local_emb):
        self.local_emb = local_emb
        local_emb_ref, local_emb_pos1, local_emb_pos2, local_emb_neg1, local_emb_neg2 = torch.tensor_split(local_emb, 5, dim=0)
        
        local_emb_ref = local_emb_ref.repeat(2, 1)
        local_emb_pos = torch.cat((local_emb_pos1, local_emb_pos2), dim=0)
        local_emb_neg = torch.cat((local_emb_neg1, local_emb_neg2), dim=0)
        
        dist_pos = self.distance(local_emb_ref, local_emb_pos)
        dist_neg = self.distance(local_emb_ref, local_emb_neg)
        
        self._sanity_check(dist_pos, dist_pos)
        
        _loss = F.relu(dist_pos - dist_neg + self.margin).mean()
        
        return _loss