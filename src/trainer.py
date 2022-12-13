import os
from tqdm.notebook import trange

import torch
import torch.nn as nn

from data_loader import *
from utils_layer import *

class Trainer:
    def __init__(self, model, loss,
                 model_dir='./', 
                ):
        self.model = model
        self.loss = loss(self.model.manifold.distance)
        self.loss.eval()
        
        # Save
        self.model_dir = model_dir + '_'.join([self.model.manifold.name, f'dim-{self.model.dim_embedding}/'])
        os.makedirs(self.model_dir + 'checkpoint/', exist_ok=True)
        
    def _compute_loss(self, loader, noise_scale, scale_band):
        batch = loader.get_batch(as_triplet=True, noise_scale=noise_scale, scale_band=scale_band)
        batch = torch.cat(batch, dim=0)
        
        local_emb = self.model(batch)
        _loss = self.loss(local_emb)
        
        return _loss
    
    def _save_model(self, epoch=None):
        if epoch is None:
            ckpt_name = 'best.pt'
        else:
            assert isinstance(epoch, int), '"epoch" is not an integer.'
            ckpt_name = f'epoch-{epoch}.pt'
        
        ckpt = {'state_dict':self.model.state_dict(), 
                'target_manifold':self.model.manifold.name,
                'dim_data':self.model.dim_data,
                'dim_embedding':self.model.dim_embedding,
                'layer_configs':self.model.layer_configs,
               }
        
        torch.save(ckpt, self.model_dir + 'checkpoint/' + ckpt_name)
        
    def fit(self, train_data, batch_size, n_epochs, valid_data=None,
            optimizer=torch.optim.Adam, optim_configs={}, 
            **kwargs
           ):
        # Keyword arguments
        noise_scale = kwargs.pop('noise_scale', 1)
        scale_band = kwargs.pop('scale_band', (0.8, 1.2))
        max_norm = kwargs.pop('max_norm', 5)
        progress_bar = kwargs.pop('progress_bar', True)
        save_every = kwargs.pop('save_every', 50)
        
        # 
        loss_history_train = []
        loader = DataLoader(train_data, batch_size)
        if valid_data is not None:
            loss_history_valid = []
            loader_valid = DataLoader(valid_data, batch_size)
            do_validation = True
            best_loss_val = 1e9
        else:
            do_validation = False
        optim = optimizer(self.model.parameters(), **optim_configs)
        
        # Epochs
        iterator = trange(n_epochs) if progress_bar else range(n_epochs)
        with open(self.model_dir + '/log.txt', 'w') as f:
            pass
        for i in iterator:
            # Training
            self.model.train()
            _loss_val = 0
            n_batches = loader.batch_indices.__len__()
            for j in range(n_batches):
                optim.zero_grad()
                _loss = self._compute_loss(loader, noise_scale, scale_band)
                _loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm, error_if_nonfinite=True)
                optim.step()
                _loss_val = _loss_val + _loss.detach().cpu()
            _loss_val = _loss_val.item() / n_batches
            loss_history_train.append(_loss_val)
            
            log_txt = f'epoch, {i}, train_loss, {_loss_val}'
            
            # Validation
            if do_validation:
                self.model.eval()
                _loss_val = 0
                n_batches = loader_valid.batch_indices.__len__()
                for _ in range(n_batches):
                    _loss_val = _loss_val + self._compute_loss(loader_valid, noise_scale, scale_band)
                _loss_val = _loss_val.detach().cpu().item() / n_batches
                loss_history_valid.append(_loss_val)
                log_txt = log_txt + f', valid_loss, {_loss_val}'
                
                if best_loss_val > _loss_val:
                    best_loss_val = _loss_val
                    self._save_model()
                
            with open(self.model_dir + '/log.txt', 'a') as f:
                f.write(log_txt + '\n')
                
            if i % save_every == 0:
                self._save_model(epoch=i)
        self._save_model(epoch=n_epochs)
        return (loss_history_train, loss_history_valid) if do_validation else loss_history_train