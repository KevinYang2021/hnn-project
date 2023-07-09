import numpy as np
import scipy
import rebound

solve_ivp = scipy.integrate.solve_ivp

import os, sys, pickle

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


def rotate2d(p, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return (R @ p.reshape(2, 1)).squeeze()


def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], nbodies=2, **kwargs):
    if not 'rtol' in kwargs:
        kwargs['rtol'] = 1e-9
    orbit_settings = locals()
    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval
    # sol = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(), t_eval=t_eval, **kwargs)
    # doesnt work for large magnitudes of difference in mass
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    for i in range(nbodies):
        sim.add(m=state[i][0], x=state[i][1], y=state[i][2], z=state[i][3], vx=state[i][4], vy=state[i][5],
                vz=state[i][6])
    sim.integrator = "whfast"
    sim.dt = 0.001
    sim.move_to_com()
    sol = {'y': np.zeros((nbodies, 7, t_points))}
    for i, t in enumerate(t_eval):
        sim.integrate(t, exact_finish_time=0)
        for j in range(nbodies):
            sol['y'][j, :, i] = [sim.particles[j].m, sim.particles[j].x, sim.particles[j].y, sim.particles[j].z,
                                 sim.particles[j].vx, sim.particles[j].vy, sim.particles[j].vz]
    sim = None
    orbit = sol['y'].reshape(nbodies, 7, t_points)
    return orbit, orbit_settings


def get_nn_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], nbodies=2, **kwargs):
    if not 'rtol' in kwargs:
        kwargs['rtol'] = 1e-9
    orbit_settings = locals()
    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval
    sol = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(), t_eval=t_eval, **kwargs)
    orbit = sol['y'].reshape(nbodies, 7, t_points)
    return orbit, orbit_settings


def sjs():
    # state = np.zeros((3, 5))
    # state[:, 0] = 1
    # p1 = 2 * np.random.rand(2) - 1
    # r = np.random.rand() * (max_radius - min_radius) + min_radius
    #
    # p1 *= r / np.sqrt(np.sum((p1 ** 2)))
    # p2 = rotate2d(p1, theta=2 * np.pi / 3)
    # p3 = rotate2d(p2, theta=2 * np.pi / 3)
    #
    # # # velocity that yields a circular orbit
    # v1 = rotate2d(p1, theta=np.pi / 2)
    # v1 = v1 / r ** 1.5
    # v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
    # v2 = rotate2d(v1, theta=2 * np.pi / 3)
    # v3 = rotate2d(v2, theta=2 * np.pi / 3)
    #
    # # make the circular orbits slightly chaotic
    # v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
    # v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
    # v3 *= 1 + nu * (2 * np.random.rand(2) - 1)
    #
    # state[0, 1:3], state[0, 3:5] = p1, v1
    # state[1, 1:3], state[1, 3:5] = p2, v2
    # state[2, 1:3], state[2, 3:5] = p3, v3
    # return state
    state = np.zeros((2, 7))
    # state[0][0] = 1e-20
    state[0][0] = 9.543e-4
    state[1][0] = 2.857e-4

    # state[0][1] = 0
    # state[0][2] = 0
    # state[0][3] = 0

    # range of sma for jupiter in au [4, 6.5]
    jsma = np.random.uniform(4, 6.5)
    # range of sma for saturn in au [8, 11]
    ssma = np.random.uniform(8, 11)

    # range of eccentricity for jupiter [0, 0.08]
    jecc = np.random.uniform(0, 0.08)
    # range of eccentricity for saturn [0, 0.1]
    secc = np.random.uniform(0, 0.1)

    # range of inclination for jupiter in degrees [0, 3]
    jinc = np.random.uniform(0, 3)
    # range of inclination for saturn in degrees [0, 5]
    sinc = np.random.uniform(0, 5)

    # range of true anomaly for jupiter in degrees [0, 360]
    jtrue = np.random.uniform(330, 360)
    # range of true anomaly for saturn in degrees [0, 360]
    strue = np.random.uniform(0, 30)

    # range of longitude of ascending node for jupiter in degrees [0, 360]
    jlong = 0
    # range of longitude of ascending node for saturn in degrees [0, 360]
    slong = 0

    # range of argument of periapsis for jupiter in degrees [0, 360]
    jargp = 0
    # range of argument of periapsis for saturn in degrees [0, 360]
    sargp = 0

    # proper symbols
    a = jsma
    e = jecc
    i = jinc
    Omega = jlong
    omega = jargp
    nu = jtrue

    # convert to radians
    i = np.deg2rad(i)
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)
    nu = np.deg2rad(nu)

    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    jx = r * (np.cos(Omega) * np.cos(omega + nu) - np.sin(Omega) * np.sin(omega + nu) * np.cos(i))
    jy = r * (np.sin(Omega) * np.cos(omega + nu) + np.cos(Omega) * np.sin(omega + nu) * np.cos(i))
    jz = r * (np.sin(i) * np.sin(omega + nu))

    # for saturn
    a = ssma
    e = secc
    i = sinc
    Omega = slong
    omega = sargp
    nu = strue

    # convert to radians
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

    # jupiter has 12.44 to 13.72 km/s or 2.62421 to 2.89423 au/yr
    # saturn has 9.14 to 10.14 km/s or 1.92808 to 2.13903 au/yr

    # range of velocity for jupiter in au/yr [2.62421, 2.89423]
    jvel = np.random.uniform(2.62421, 2.89423)
    # range of velocity for saturn in au/yr [1.92808, 2.13903]
    svel = np.random.uniform(1.92808, 2.13903)

    # set velocity tangent to the position vector (perpendicular to the radius)

    # jupiter
    jvx = -jy * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvy = jx * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvz = 0

    # saturn
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


def sample_orbits(timesteps=20, samples=2000, nbodies=2, t_span=[0, 20], verbose=True, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    e = []
    N = timesteps * samples
    while len(x) < N:
        state = sjs()
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        batch = orbit.transpose(2, 0, 1).reshape(-1, nbodies * 7)

        if verbose:
            if len(x) % 1000 == 0:
                print("Generated {} / {} orbits".format(len(x), N))

        for state in batch:
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
