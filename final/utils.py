import torch, pickle, os
import numpy as np


def rk4(fun, y0, t, dt, *args, **kwargs):
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt / 2.0 * k1, t + dt / 2.0, *args, **kwargs)
    k3 = fun(y0 + dt / 2.0 * k2, t + dt / 2.0, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)

    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl

def from_pickle(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def to_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def mse_loss(u, v):
    squared = (u - v) / u.shape[0]
    squared = squared ** 2
    sum = torch.sum(squared)
    loss = sum * u.shape[0]
    return loss
