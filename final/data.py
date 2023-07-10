import numpy as np
import scipy
import rebound

import os, sys, pickle
from utils import from_pickle, to_pickle

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


def potential_energy(state):
    total = np.zeros((1, 1, state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            r_ij = ((state[i:i + 1, 1:4] - state[j:j + 1, 1:4]) ** 2).sum(1, keepdims=True) ** .5
            m_i = state[i:i + 1, 0:1]
            m_j = state[j:j + 1, 0:1]
            total += m_i * m_j / r_ij
    U = -total.sum(0).squeeze()
    return U


def kinetic_energy(state):
    energies = .5 * state[:, 0:1] * (state[:, 4:7] ** 2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T


def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


def get_accelerations(state, epsilon=0):
    net_accs = []
    for i in range(state.shape[0]):
        other_bodies = np.concatenate([state[:i, :], state[i + 1:, :]], axis=0)
        displacements = other_bodies[:, 1:4] - state[i, 1:4]
        distances = (displacements ** 2).sum(1, keepdims=True) ** 0.5
        masses = other_bodies[:, 0:1]
        pointwise_accs = masses * displacements / (distances ** 3 + epsilon)
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs


def update(t, state):
    state = state.reshape(-1, 7)
    deriv = np.zeros_like(state)
    deriv[:, 1:4] = state[:, 4:7]
    deriv[:, 4:7] = get_accelerations(state)
    return deriv.reshape(-1)


def sjs():
    state = np.zeros((2, 7))
    state[0][0] = 9.543e-4
    state[1][0] = 2.857e-4

    jsma = np.random.uniform(4, 6.5)
    ssma = np.random.uniform(8, 11)
    jecc = np.random.uniform(0, 0.08)
    secc = np.random.uniform(0, 0.1)
    jinc = np.random.uniform(0, 3)
    sinc = np.random.uniform(0, 5)
    jtrue = np.random.uniform(0, 360)
    strue = np.random.uniform(0, 360)
    jlong = 0
    slong = 0
    jargp = 0
    sargp = 0

    a = jsma
    e = jecc
    i = jinc
    Omega = jlong
    omega = jargp
    nu = jtrue

    i = np.deg2rad(i)
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)
    nu = np.deg2rad(nu)

    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    jx = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    jy = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    jz = r * (np.sin(i) * np.sin(omega + nu))

    a = ssma
    e = secc
    i = sinc
    Omega = slong
    omega = sargp
    nu = strue

    i = np.deg2rad(i)
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)
    nu = np.deg2rad(nu)

    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    sx = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    sy = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    sz = r * (np.sin(i) * np.sin(omega + nu))

    state[0][1] = jx
    state[0][2] = jy
    state[0][3] = jz

    state[1][1] = sx
    state[1][2] = sy
    state[1][3] = sz

    jvel = np.random.uniform(2.62421, 2.89423)
    svel = np.random.uniform(1.92808, 2.13903)

    jvx = -jy * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvy = jx * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvz = 0

    svx = -sy * svel / np.sqrt(sx ** 2 + sy ** 2)
    svy = sx * svel / np.sqrt(sx ** 2 + sy ** 2)
    svz = 0

    state[0][4] = jvx
    state[0][5] = jvy
    state[0][6] = jvz
    state[1][4] = svx
    state[1][5] = svy
    state[1][6] = svz

    return state


def sample_orbits(timesteps=60, samples=50000, nbodies=2, t_span=[0, 20], verbose=True, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    e = []
    N = timesteps * samples
    while len(x) < N:
        state = sjs()

        if verbose:
            if len(x) % 1000 == 0:
                print("Generated {} / {} orbits".format(len(x), N))

        dstate = update(None, state)
        coords = state.reshape(nbodies, 7).T[1:].flatten()
        x.append(coords)
        dcoords = dstate.reshape(nbodies, 7).T[1:].flatten()
        dx.append(dcoords)
        shaped_state = state.copy().reshape(nbodies, 7, 1)
        e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N], 'dcoords': np.stack(dx)[:N], 'energy': np.stack(e)[:N]}
    return data, orbit_settings


def make_dataset(test_split=0.15, **kwargs):
    data, settings = sample_orbits(**kwargs)
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data
    data['meta'] = settings
    return data, settings


def get_dataset(name, save_dir, **kwargs):
    path = '{}/{}.pkl'.format(save_dir, name)
    try:
        data = from_pickle(path)
        print("Loaded dataset from {}".format(path))
    except:
        data, settings = make_dataset(**kwargs)
        to_pickle(data, path)
        print("Saved dataset to {}".format(path))

    return data
