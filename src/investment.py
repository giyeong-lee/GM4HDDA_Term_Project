import os
from tqdm.notebook import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioWeight(nn.Module):
    def __init__(self, encoder, num_currencies):
        super().__init__()
        
        self.encoder = encoder
        in_features = encoder.dim_embedding
        
        self.FF11 = nn.Linear(in_features, in_features)
        self.FF12 = nn.Linear(in_features, 128)
        self.LN1 = nn.LayerNorm(128)
        
        self.FF21 = nn.Linear(128, 128)
        self.FF22 = nn.Linear(128, 128)
        self.LN2 = nn.LayerNorm(128)
        
        # last two_channel : time, cash channel added
        self.Out = nn.Linear(128, num_currencies+1)
        
        self.device = encoder.device
        self.to(encoder.device)
        
    def forward(self, x):
        x = self.encoder(x.to(self.device))
        x = x + F.relu(self.FF11(x))
        x = self.LN1(self.FF12(x))
        
        x = x + F.relu(self.FF21(x))
        x = self.LN2(self.FF22(x))
        
        x = self.Out(x)
        
        W = F.softmax(x, dim=-1)
        
        return W

class Investment:
    def __init__(self, encoder, num_currencies, init_capital=1e7):
        self.init_capital = init_capital
        self.rebalance = PortfolioWeight(encoder, num_currencies)
        self.num_currencies = num_currencies
    
    def _trade(self, episode, ):
        device = self.rebalance.device
        episode_length = episode.shape[0]-1
        hold = torch.zeros((self.num_currencies+1)).to(device)
        hold[0] = self.init_capital
        position_values = torch.zeros(episode_length+1, ).to(device)
        position_values[0] = self.init_capital
        portfolios = torch.zeros((episode_length, self.num_currencies+1))
        
        episode = episode.to(device)
        p_weights = self.rebalance(episode)
        
        for j in range(episode_length):
            p_weight = p_weights[j]
            
            cur_price = episode[j, -1, :-2].squeeze()
            cur_position = torch.cat((hold[[0]], cur_price*hold[1:]))
            cur_position_value = cur_position.sum()

            # Record current choice
            portfolios[j] = p_weight.detach().cpu()

            new_position = p_weight * cur_position_value
            hold = (new_position / torch.cat((torch.ones(1, ).to(device), cur_price))).squeeze()

            next_price = episode[j+1, -1, :-2].to(device)
            market_change = cur_price - next_price
            next_position_value = cur_position_value + (hold[1:] * market_change).sum()
            
            # Record performance
            position_values[j+1] = next_position_value

        profit_loss = position_values[1:] / position_values[:-1] - 1
        neg_sharpe_ratio = - profit_loss.mean() / profit_loss.std()
        
        return neg_sharpe_ratio, position_values, portfolios
        
    def fit(self, train_data, valid_data, n_epochs, episode_length, 
            optimizer=torch.optim.Adam, optim_configs={}, save_name='investment.pt', **kwargs):
        # Keyword arguments
        progress_bar = kwargs.pop('progress_bar', True)
        
        optim = optimizer(self.rebalance.parameters(), **optim_configs)
        device = self.rebalance.device
        
        prev_max_sharpe = -1e7
        
        #
        iterator = trange(n_epochs) if progress_bar else range(n_epochs)
        N_train = train_data.shape[0]
        for i in iterator:
            self.rebalance.train()
            start_idx = torch.randperm(N_train-episode_length-1)[0].item()
            optim.zero_grad()
            _loss, _, _ = self._trade(train_data[start_idx:(start_idx+episode_length+1)])
            _loss.backward()
            optim.step()
            
            # Validation
            self.rebalance.eval()
            neg_sharpe_ratio, position_values, portfolios = self._trade(valid_data)
            sharpe_ratio = - neg_sharpe_ratio.detach().cpu()
            if sharpe_ratio > prev_max_sharpe:
                torch.save(self.rebalance.state_dict(), './results/investment/models/' + save_name)