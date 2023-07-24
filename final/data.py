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
    state = state.reshape(-1, 4)
    deriv = np.zeros_like(state)
    deriv[:, 1:4] = get_accelerations(state)
    return deriv.reshape(-1)


def kep2cart(a, e, i, Omega, omega, nu):
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    return np.array([r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i)),
                     r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i)),
                     r * (np.sin(i) * np.sin(omega + nu))])


def sjs():
    state = np.zeros((2, 4))
    state[0][0] = 9.543e-4
    state[1][0] = 2.857e-4

    jsma = np.random.uniform(4, 6.5)
    ssma = np.random.uniform(8, 11)
    jecc = np.random.uniform(0, 0.08)
    secc = np.random.uniform(0, 0.1)
    jinc = np.random.uniform(0, 3)
    sinc = np.random.uniform(0, 5)
    jtrue = np.random.uniform(0, 2 * np.pi)
    strue = np.random.uniform(0, 2 * np.pi)
    jlong = 0
    slong = 0
    jargp = 0
    sargp = 0
    state[0][1:4] = kep2cart(jsma, jecc, jinc, jlong, jargp, jtrue)
    state[1][1:4] = kep2cart(ssma, secc, sinc, slong, sargp, strue)
    return state


def gen_orbital_elements(sma, ecc, inc, true, long, argp):
    return np.random.uniform(sma[0], sma[1]), np.random.uniform(ecc[0], ecc[1]), np.deg2rad(
        np.random.uniform(inc[0], inc[1])), np.deg2rad(np.random.uniform(true[0], true[1])), np.deg2rad(
        np.random.uniform(long[0], long[1])), np.deg2rad(np.random.uniform(argp[0], argp[1]))


def solarsys():
    state = np.zeros((8, 4))
    state[0][0] = 1.660e-07
    state[1][0] = 2.448e-06
    state[2][0] = 3.004e-06
    state[3][0] = 3.227e-07
    state[4][0] = 9.543e-04
    state[5][0] = 2.857e-04
    state[6][0] = 4.366e-05
    state[7][0] = 5.151e-05

    a, e, i, true, long, argp = gen_orbital_elements([0.3, 0.5], [0.0, 0.2], [0, 8], [0, 360], [0, 0], [0, 0])
    state[0][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([0.6, 0.8], [0.0, 0.02], [0, 4], [0, 360], [0, 0], [0, 0])
    state[1][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([0.9, 1.1], [0.0, 0.04], [0, 1], [0, 360], [0, 0], [0, 0])
    state[2][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([1.2, 1.4], [0.0, 0.06], [0, 1], [0, 360], [0, 0], [0, 0])
    state[3][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([4, 6.5], [0.0, 0.08], [0, 3], [0, 360], [0, 0], [0, 0])
    state[4][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([8, 11], [0.0, 0.1], [0, 5], [0, 360], [0, 0], [0, 0])
    state[5][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([17, 21], [0.0, 0.08], [0, 1], [0, 360], [0, 0], [0, 0])
    state[6][1:4] = kep2cart(a, e, i, long, argp, true)
    a, e, i, true, long, argp = gen_orbital_elements([28, 32], [0.0, 0.01], [0, 4], [0, 360], [0, 0], [0, 0])
    state[7][1:4] = kep2cart(a, e, i, long, argp, true)

    return state


def sample_sjs(timesteps=60, samples=50000, nbodies=2, t_span=[0, 20], verbose=True, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    # e = []
    N = timesteps * samples
    while len(x) < N:
        state = solarsys()

        if verbose:
            if len(x) % 1000 == 0:
                print("Generated {} / {} orbits".format(len(x), N))

        dstate = update(None, state)
        coords = state.reshape(nbodies, 4).T[1:].flatten()
        x.append(coords)
        dcoords = dstate.reshape(nbodies, 4).T[1:].flatten()
        dx.append(dcoords)
        # shaped_state = state.copy().reshape(nbodies, 7, 1)
        # e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N], 'dcoords': np.stack(dx)[:N]}
    # data = {'coords': np.stack(x)[:N], 'dcoords': np.stack(dx)[:N], 'energy': np.stack(e)[:N]}
    return data, orbit_settings


def sample_solarsys(timesteps=60, samples=50000, nbodies=8, t_span=[0, 20], verbose=True, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    # e = []
    N = timesteps * samples
    while len(x) < N:
        state = solarsys()

        if verbose:
            if len(x) % 1000 == 0:
                print("Generated {} / {} orbits".format(len(x), N))

        dstate = update(None, state)
        coords = state.reshape(nbodies, 4).T[1:].flatten()
        x.append(coords)
        dcoords = dstate.reshape(nbodies, 4).T[1:].flatten()
        dx.append(dcoords)

    data = {'coords': np.stack(x)[:N], 'dcoords': np.stack(dx)[:N]}
    return data, orbit_settings


def make_dataset(name, test_split=0.15, **kwargs):
    if (name == 'sjs'):
        data, settings = sample_sjs(**kwargs)
    elif (name == 'solarsys'):
        data, settings = sample_solarsys(**kwargs)
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
        data, settings = make_dataset(name, **kwargs)
        to_pickle(data, path)
        print("Saved dataset to {}".format(path))

    return data
