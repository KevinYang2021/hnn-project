import autograd
import autograd.numpy as np
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp

import matplotlib.pyplot as plt


def hamiltonian(coords):
    q, p = np.split(coords, 2)
    H = 9.81 * (1 - np.cos(q)) + (p ** 2) * 0.5
    return H


def dynamics(t, coords):
    dc = autograd.grad(hamiltonian)(coords)
    dq, dp = np.split(dc, 2)
    S = np.concatenate([dp, -dq], axis=-1)
    return S


def get_trajectory(t_span=[0, 3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial state
    if y0 is None:
        y0 = np.random.rand(2) * 2. - 1
    if radius is None:
        radius = np.random.rand() + 1.3  # sample a range of radii
    y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius  ## set the appropriate radius

    spring_ivp = solve_ivp(fun=dynamics, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt, 2)

    # add noise
    q += np.random.randn(*q.shape) * noise_std
    p += np.random.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t_eval


def get_dataset(seed=0, samples=100, test_split=0.5, **kwargs):
    # data={'meta':locals()}
    data = {}

    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(np.stack([x, y]).T)
        dxs.append(np.stack([dx, dy]).T)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]

    data = split_data
    return data


def get_field(xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0, gridsize=20):
    field = {'meta': locals()}
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    y = np.stack([a.flatten(), b.flatten()])
    dy = [dynamics(None, y) for y in y.T]
    dy = np.stack(dy).T

    field['x'] = y.T
    field['dx'] = dy.T
    return field
