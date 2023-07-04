import torch
import numpy as np


def rk4(fun, y0, t, dt, *args, **kwargs):
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt / 2.0 * k1, t + dt / 2.0, *args, **kwargs)
    k3 = fun(y0 + dt / 2.0 * k2, t + dt / 2.0, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)

    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


class hnn(torch.nn.Module):
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal', assume_canonical_coords=True):
        super(hnn, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.field_type = field_type

    def forward(self, x):
        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1, 1)

    def rk4_time_deriv(self, x, dt):
        return rk4(fun=self.time_deriv, y0=x, t=0, dt=dt)

    def time_deriv(self, x, t=None, separate_fields=False):
        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

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
