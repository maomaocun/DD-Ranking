import torch.nn.functional as F
import numpy as np
import torch
from math import ceil
import torch .nn as nn
import os
# 加载合成数据
from models import define_model
def load_synthetic_data(data_dir, config,use_softlabel=False):
    config.config["ipc"]= config.config["ipc"] *4
    """加载合成数据并解码"""
    data = torch.load(data_dir, map_location='cpu')
    syn_images = data[0]
    syn_labels = data[1]
    data_dec = []
    target_dec = []

    # 按类别分离数据并解码
    for c in range(0, config.get('nclass')):
        target_mask = syn_labels == c
        data = syn_images[target_mask].detach()
        target = syn_labels[target_mask].detach()
        data, target = decode(data, target,size=config.get('im_size'))
        data_dec.append(data)
        target_dec.append(target)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)
    if use_softlabel:
        tearcher_model = define_model(config.get("dataset"),config.get('model_name'),config.get('nclass'), config.get('im_size')[0])
    print(f"Decode condensed data: {data_dec.shape},label: {target_dec.shape}")

    return data_dec, target_dec

def decode(data,target,decode_type="single",size=(32,32),factor=2, bound=10000):

    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(size,data, target, factor)
        elif decode_type == 'bound':
            data, target = decode_zoom_bound(size,data, target, factor, bound=bound)
        else:
            data, target = decode_zoom(size,data, target, factor)

    return data, target


def subsample(data, target, max_size=-1):
    if (data.shape[0] > max_size) and (max_size > 0):
        indices = np.random.permutation(data.shape[0])
        data = data[indices[:max_size]]
        target = target[indices[:max_size]]

    return data, target

def decode_zoom(size, img, target, factor):
    resizor = nn.Upsample(size=size, mode='bilinear') 
    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor): 
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resizor(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec

def decode_zoom_multi(size, img, target, factor_max):
    """Multi-scale multi-formation
    """
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(size,img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)

def decode_zoom_bound(size, img, target, factor_max, bound=128):
    bound_cur = bound - len(img)
    budget = len(img)

    data_multi = []
    target_multi = []

    idx = 0
    decoded_total = 0
    for factor in range(factor_max, 0, -1):
        decode_size = factor**2
        if factor > 1:
            n = min(bound_cur // decode_size, budget)
        else:
            n = budget

        decoded = decode_zoom(size,img[idx:idx + n], target[idx:idx + n], factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])
        idx += n
        budget -= n
        decoded_total += n * decode_size
        bound_cur = bound - decoded_total - budget

        if budget == 0:
            break

    data_multi = torch.cat(data_multi)
    target_multi = torch.cat(target_multi)
    return data_multi, target_multi
