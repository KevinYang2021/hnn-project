import torch
import numpy as np
from MLP import MLP


def rk4(fun, y0, t, dt, *args, **kwargs):
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt / 2.0 * k1, t + dt / 2.0, *args, **kwargs)
    k3 = fun(y0 + dt / 2.0 * k2, t + dt / 2.0, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)

    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


class nn_model(torch.nn.Module):
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal', assume_canonical_coords=True):
        super(nn_model, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.field_type = field_type

    def forward(self, x):
        return self.differentiable_model(x)

    def rk4_time_deriv(self, x, dt):
        return rk4(fun=self.time_deriv, y0=x, t=0, dt=dt)

    def time_deriv(self, x, t=None, separate_fields=False):
        return self.differentiable_model(x)

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n // 2:], -M[:n // 2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n, n)  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M
