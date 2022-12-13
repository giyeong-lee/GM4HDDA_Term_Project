import torch
import torch.nn as nn
import torch.nn.functional as F

# Euclidean Space
class EuclideanSpace(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_local_coords = dim
        self.dim_manifold = dim
        self.name = 'Euclidean'
    
    def forward(self, x):
        local_coords = x
        
        return local_coords
    
    def local_to_manifold(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        points = local_coords
        
        return points
    
    def metric(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        H = torch.eye(d).view(1, d, d).repeat(N, 1, 1).to(local_coords)
        
        return H
    
    def distance(self, local_coords1, local_coords2):
        N, d1 = local_coords1.shape
        N, d2 = local_coords2.shape
        self._dimension_check(d1, 'local_coords')
        self._dimension_check(d2, 'local_coords')
        
        # p = local_coords
        dist = (local_coords1-local_coords2).norm(dim=-1)
        
        return dist
    
    def _dimension_check(self, dim, to_check):
        if to_check == 'local_coords':
            dim_match = self.dim_local_coords
        else:
            dim_match = self.dim_manifold
            
        assert dim == dim_match, f'Dimension Mismatch -- Input: {dim}, {to_check.title()}: {dim_match}'
        
# Hypersphere - Spherical Charts
class Hypersphere(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_local_coords = dim
        self.dim_manifold = dim+1
        
        self._some_constants()
        
        self.name = 'Hypersphere'
    
    def forward(self, x):
        '''
        R^d -> (0, pi)^{d-1} x (0, 2pi)
        '''
        local_coords = (torch.arctan(x) + torch.pi/2) * self.scale.to(x)
        
        return local_coords
    
    def local_to_manifold(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        '''
        (0, pi)^{d-1} x (0, 2pi) -> S^d
        '''
        arr = torch.arange(d)
        points = torch.ones((N, d+1, d)).to(local_coords)
        points[:, arr, arr] = torch.cos(local_coords)
        points[:, self.idx[0], self.idx[1]] = torch.sin(local_coords[:, self.idx[1]])
        points = points.prod(dim=-1)
        
        # Just for numerical stability
        points = points / points.norm(dim=-1).unsqueeze(1)
        
        return points
    
    def metric(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        s = torch.cat((torch.ones((N, 1)).to(local_coords), 
                       torch.sin(local_coords[:, :-1]).square().cumprod(dim=-1)), 
                      dim=-1
                     )
        
        H = torch.eye(d).repeat(N, 1, 1).to(local_coords) * s.unsqueeze(1)
        
        return H
    
    def distance(self, local_coords1, local_coords2, eps=1e-6):
        N, d1 = local_coords1.shape
        N, d2 = local_coords2.shape
        self._dimension_check(d1, 'local_coords')
        self._dimension_check(d2, 'local_coords')
        
        points1 = self.local_to_manifold(local_coords1)
        points2 = self.local_to_manifold(local_coords2)
        dot_prod = (points1*points2).sum(dim=-1)
        
        # Just for numerical stability
        sign = dot_prod.sign()
        dot_prod = (1-eps - F.relu(1-eps - dot_prod.abs())) * sign
        dist = torch.arccos(dot_prod)
        
        return dist
    
    def _dimension_check(self, dim, to_check):
        if to_check == 'local_coords':
            dim_match = self.dim_local_coords
        else:
            dim_match = self.dim_manifold
            
        assert dim == dim_match, f'Dimension Mismatch -- Input: {dim}, {to_check.title()}: {dim_match}'
        
    def _some_constants(self, ):
        idx = torch.cat(torch.meshgrid(torch.arange(self.dim_manifold), 
                                       torch.arange(self.dim_local_coords), 
                                       indexing='ij'))
        idx = idx.reshape(2, -1)
        idx = idx[:, idx[0, :] > idx[1, :]]
        self.idx = idx
        self.mask = torch.ones((1, self.dim_manifold, self.dim_manifold), requires_grad=False)
        self.mask[:, idx[0], idx[1]] = 0
        self.mask = self.mask.transpose(1, 2)
        self.scale = torch.ones((1, self.dim_local_coords), requires_grad=False)
        self.scale[0:, -1] += 1
    
# Hypersphere - Stereographic Projection Charts
class HypersphereStereographic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_local_coords = dim
        self.dim_manifold = dim+1
        
        self.name = 'Hypersphere_Stereographic'
    
    def forward(self, x):
        local_coords = x
        
        return local_coords
    
    def local_to_manifold(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        '''
        R^d -> S^d
        '''
        norm_sq = local_coords.norm(dim=-1).square().view(-1, 1)
        points = torch.cat((2*local_coords / (norm_sq+1), (norm_sq-1) / (norm_sq+1)), dim=-1)
        
        return points
    
    def metric(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        s = 4 / (local_coords.norm(dim=-1).square().view(-1, 1)+1).square()
        
        H = torch.eye(d).repeat(N, 1, 1).to(local_coords) * s
        
        return H
    
    def distance(self, local_coords1, local_coords2, eps=1e-6):
        N, d1 = local_coords1.shape
        N, d2 = local_coords2.shape
        self._dimension_check(d1, 'local_coords')
        self._dimension_check(d2, 'local_coords')
        
        points1 = self.local_to_manifold(local_coords1)
        points2 = self.local_to_manifold(local_coords2)
        
        dot_prod = (points1*points2).sum(dim=-1)
        
        # Just for numerical stability
        sign = dot_prod.sign()
        dot_prod = (1-eps - F.relu(1-eps - dot_prod.abs())) * sign
        dist = torch.arccos(dot_prod)
        
        return dist
        
    def _dimension_check(self, dim, to_check):
        if to_check == 'local_coords':
            dim_match = self.dim_local_coords
        else:
            dim_match = self.dim_manifold
            
        assert dim == dim_match, f'Dimension Mismatch -- Input: {dim}, {to_check.title()}: {dim_match}'
        
    
# Poincare Half Plane
class PoincareHalfplane(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_local_coords = dim
        self.dim_manifold = dim
    
        self.name = 'Poincare-Halfplane'
    
    def forward(self, x):
        local_coords = x
        
        return local_coords
    
    def local_to_manifold(self, local_coords):
        '''
        R^d -> R^{d-1} x (0, inf)
        '''
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        points = torch.cat((local_coords[:, :-1], F.softplus(local_coords[:, [-1]])), dim=-1)
        
        return points
    
    def metric(self, local_coords):
        N, d = local_coords.shape
        self._dimension_check(d, 'local_coords')
        
        points = self.local_to_manifold(local_coords)
        
        H = torch.eye(d).repeat(N, 1, 1).to(points)
        H = H / points[:, [-1]].square().unsqueeze(2).repeat(1, d, d)
        
        return H
    
    def distance(self, local_coords1, local_coords2):
        N, d1 = local_coords1.shape
        N, d2 = local_coords2.shape
        self._dimension_check(d1, 'local_coords')
        self._dimension_check(d2, 'local_coords')
        
        points1 = self.local_to_manifold(local_coords1)
        points2 = self.local_to_manifold(local_coords2)
        
        dist = (points1-points2).norm(dim=-1) / (2*(points1[:, -1]*points2[:, -1]).sqrt())
        dist = 2*torch.arcsinh(dist)
        
        return dist
    
    def _dimension_check(self, dim, to_check):
        if to_check == 'local_coords':
            dim_match = self.dim_local_coords
        else:
            dim_match = self.dim_manifold
            
        assert dim == dim_match, f'Dimension Mismatch -- Input: {dim}, {to_check.title()}: {dim_match}'
        