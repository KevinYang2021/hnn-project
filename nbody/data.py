import numpy as np
import scipy

solve_ivp = scipy.integrate.solve_ivp

import os, sys, pickle

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


##### ENERGY #####
def potential_energy(state):
    total = np.zeros((1, 1, state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            r_ij = ((state[i:i + 1, 1:3] - state[j:j + 1, 1:3]) ** 2).sum(1, keepdims=True) ** .5
            m_i = state[i:i + 1, 0:1]
            m_j = state[j:j + 1, 0:1]
            total += m_i * m_j / r_ij
    U = -total.sum(0).squeeze()
    return U


def kinetic_energy(state):
    energies = .5 * state[:, 0:1] * (state[:, 3:5] ** 2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T


def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


def get_accelerations(state, epsilon=0):
    net_accs = []
    for i in range(state.shape[0]):
        other_bodies = np.concatenate([state[:i, :], state[i + 1:, :]], axis=0)
        displacements = other_bodies[:, 1:3] - state[i, 1:3]
        distances = (displacements ** 2).sum(1, keepdims=True) ** 0.5
        masses = other_bodies[:, 0]
        pointwise_accs = masses * displacements / (distances ** 3 + epsilon)
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs


def update(t, state):
    state = state.reshape(-1, 5)
    deriv = np.zeros_like(state)
    deriv[:, 1:3] = state[:, 3:5]
    deriv[:, 3:5] = get_accelerations(state)
    return deriv.reshape(-1)


def rotate2d(p, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return (R @ p.reshape(2, 1)).squeeze()


def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], nbodies=3, **kwargs):
    if not 'rtol' in kwargs:
        kwargs['rtol'] = 1e-9
    orbit_settings = locals()
    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval
    sol = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(), t_eval=t_eval, **kwargs)
    orbit = sol['y'].reshape(nbodies, 5, t_points)
    return orbit, orbit_settings


def random_config(nu, min_radius, max_radius):
    state = np.zeros((3, 5))
    state[:, 0] = 1
    p1 = 2 * np.random.rand(2) - 1
    r = np.random.rand() * (max_radius - min_radius) + min_radius

    p1 *= r / np.sqrt(np.sum((p1 ** 2)))
    p2 = rotate2d(p1, theta=2 * np.pi / 3)
    p3 = rotate2d(p2, theta=2 * np.pi / 3)

    # # velocity that yields a circular orbit
    v1 = rotate2d(p1, theta=np.pi / 2)
    v1 = v1 / r ** 1.5
    v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
    v2 = rotate2d(v1, theta=2 * np.pi / 3)
    v3 = rotate2d(v2, theta=2 * np.pi / 3)

    # make the circular orbits slightly chaotic
    v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
    v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
    v3 *= 1 + nu * (2 * np.random.rand(2) - 1)

    state[0, 1:3], state[0, 3:5] = p1, v1
    state[1, 1:3], state[1, 3:5] = p2, v2
    state[2, 1:3], state[2, 3:5] = p3, v3
    return state


def sample_orbits(timesteps=20, trials=10000, nbodies=3, orbit_noise=0.2, min_radius=1, max_radius=1.5, t_span=[0, 5],
                  verbose=False, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    e = []
    N = timesteps * trials
    while len(x) < N:
        state = random_config(orbit_noise, min_radius, max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        batch = orbit.transpose(2, 0, 1).reshape(-1, nbodies * 5)

        for state in batch:
            dstate = update(None, state)
            coords = state.reshape(nbodies, 5).T[1:].flatten()
            x.append(coords)
            dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
            dx.append(dcoords)
            shaped_state = state.copy().reshape(nbodies, 5, 1)
            e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N], 'dcoords': np.stack(dx)[:N], 'energy': np.stack(e)[:N]}
    return data, orbit_settings


def make_dataset(test_split=0.2, **kwargs):
    data, settings = sample_orbits(**kwargs)
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data
    data['meta'] = settings
    return data, settings


def from_pickle(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def to_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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
