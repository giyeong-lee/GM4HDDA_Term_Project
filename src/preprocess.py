import os
import warnings

from datetime import datetime

import numpy as np
import pandas as pd

import torch
from sklearn.impute import KNNImputer

from utils_preprocess import *

def load_data(source, start_date, end_date='2022-12-31'):
    if isinstance(source, str):
        source = pd.read_csv(source).set_index('Date')
    
    date_range = list(map(lambda t: t.strftime('%Y-%m-%d'), 
                      pd.date_range(start_date, end_date)))
    date_range = sorted(list(set(date_range) & set(source.index.tolist())))
    
    return source.loc[date_range]
    
def cleanse_data(data, 
                 exclude_currency=['AED', 'USD'], nan_threshold=.01, n_neighbors=7, 
                 save_dir=None,
                ):
    # Merge EUR
    cols = list(set(map(lambda c: 'EUR' if 'EUR' in c else c, data.columns.tolist())))
    data = data[cols]
                
    # Weekends & Holidays
    data = data.loc[list(map(is_weekday, data.index)), :]
    data = data.loc[[bool(1-is_holiday(date)) for date in data.index], :]
    
    # Discarding currencies with too many NaNs
    n_data = data.shape[0]
    nan_by_currency = data.isna().sum(axis=0).values
    currency_to_keep = data.columns[nan_by_currency <= nan_threshold*n_data]
    currency_to_keep = set(currency_to_keep) - set(exclude_currency)
    currency_to_keep = sorted(list(currency_to_keep))
    data = data[currency_to_keep]
    
    # Imputation
    if data.isna().values.sum():
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        data = pd.DataFrame(imputer.fit_transform(data.values), 
                            columns=data.columns, 
                            index=data.index
                           )
    
    # Scaling by digits
    median = data.describe().loc[['50%']]
    digits = median.applymap(lambda x: int(np.log10(x)))
    data = data / (10 ** (digits.values+1))
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'
        data.to_csv(save_dir + 'cleansed_data.csv', index=False)
        
    return data
    
def date_positional_encoding(dates):
    dates = pd.to_datetime(dates)
    day_of_Y = torch.Tensor(dates.dayofyear)
    day_of_M = torch.Tensor(dates.day)
    
    pos_in_Y = day_of_Y / (torch.ones_like(day_of_Y) * 365 + dates.is_leap_year)
    pos_in_M = day_of_M / torch.Tensor(dates.daysinmonth)
    
    pos = torch.cat((pos_in_Y.unsqueeze(1), pos_in_M.unsqueeze(1)), 
                    dim=1
                   )
    
    return pos.unsqueeze(0)
    
def extract_data(data, n_timestamps, lag, 
                 save_dir=None,
                ):
    width = n_timestamps*lag
    n_data = data.shape[0]
    
    extracted_data = []
    extracted_dates = []
    extracted_pos_enc = None
    
    for i in range(width-1, n_data):
        idx = np.arange(i, i-width, -lag)[::-1]
        window = data.iloc[idx]
        
        extracted_data.append(window.values.tolist())
        extracted_dates.append(window.index[-1])
        if extracted_pos_enc is None:
            extracted_pos_enc = date_positional_encoding(window.index)
        else:
            extracted_pos_enc = torch.cat((extracted_pos_enc, date_positional_encoding(window.index)), 
                                          dim=0
                                         )
    
    extracted_data = torch.Tensor(extracted_data)
    extracted_dates = np.array(extracted_dates)
    
    extracted = {'data':extracted_data,
                 'pos_enc':extracted_pos_enc, 
                 'dates':extracted_dates
                }
                
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'
        torch.save(extracted, save_dir + 'extracted_data.pt')
        
    return extracted
    
def split_data(extracted, ratio=(.64, .16, .2), save_dir=None):
    assert sum(ratio) == 1, 'Check the ratio.'
    
    n_data = extracted['data'].shape[0]
    bdry = [0, int(n_data * (ratio[0])), int(n_data * (1-ratio[2])), n_data]
    
    split = {k1:{k2:None for k2 in ['data', 'pos_enc', 'dates']} 
             for k1 in ['train', 'valid', 'test']}
             
    for i, k in enumerate(split.keys()):
        split[k]['data'] = extracted['data'][bdry[i]:bdry[i+1]]
        split[k]['pos_enc'] = extracted['pos_enc'][bdry[i]:bdry[i+1]]
        split[k]['dates'] = extracted['dates'][bdry[i]:bdry[i+1]]
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_dir[-1] != '/':
            save_dir = save_dir + '/'
        torch.save(split, save_dir + 'data_split.pt')
        
    return split