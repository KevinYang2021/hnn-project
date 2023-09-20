import matplotlib.pyplot as plt
import torch, time, os, sys
import numpy as np
import scipy.integrate, scipy.ndimage
from nn_models import MLP
from hnn import hnn
from data import get_dataset, solarsys, potential_energy, kinetic_energy, total_energy, get_accelerations
from asteroids import asteroids
import rebound


def get_args(solarsys, asteroids):
    if solarsys:
        return {'nonlinearity': 'tanh', 'hidden_dim': 300, 'input_dim': 24, 'learning_rate': 0.001, 'batch_size': 20000,
                'total_steps': 2000, 'print_every': 1000, 'name': 'solarsys', 'verbose': True,
                'field_type': 'solenoidal', 'seed': 0, 'save_dir': '/storage/models'}
    else:
        return {'nonlinearity': 'tanh', 'hidden_dim': 300, 'input_dim': asteroids * 3, 'learning_rate': 0.001,
                'batch_size': 20000, 'total_steps': 2000, 'print_every': 1000, 'name': '{}asteroids'.format(asteroids),
                'verbose': True, 'field_type': 'solenoidal', 'seed': 0, 'save_dir': '/storage/models'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def load_model(args):
    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = hnn(args.input_dim, differentiable_model=nn_model, field_type=args.field_type)
    label = 'hnn'
    path = '{}/{}_{}.pt'.format(args.save_dir, args.name, label)
    model.load_state_dict(torch.load(path))
    return model


def velocity(min, max, pos):
    x, y, z = pos
    vel = np.random.uniform(min, max)
    vx = -y * vel / np.sqrt(x ** 2 + y ** 2)
    vy = x * vel / np.sqrt(x ** 2 + y ** 2)
    vz = 0
    return np.array([vx, vy, vz])


def get_paccelerations(t, input, model):
    sz = input.shape[0]
    input = input.reshape(-1, 4)
    deriv = np.zeros_like(input)
    np_x = input[:, 1:]  # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_deriv(x)
    deriv[:, 1:] = dx_hat.detach().data.numpy().reshape(3, sz).T
    return deriv.reshape(-1)


def get_next(t, input, pmodel, amodel):
    paccelerations = get_paccelerations(None, input[1:9, 0:4], pmodel)
    paccelerations = paccelerations.reshape(8, 4)
    paccelerations = paccelerations[:, 1:].reshape(-1).reshape((8, 3))
    paccelerations = paccelerations * G
    dx = np.zeros_like(input)
    for i in range(0, nbodies):
        dx[i][0] = 0
    dx[:, 1:4] = input[:, 4:7]
    dx[0, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    for i in range(1, 9):
        r = input[i, 1:4] - input[0, 1:4]
        r = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
        a = -G * input[0, 0] * (input[i, 1:4] - input[0, 1:4]) / r ** 3
        paccelerations[i - 1, :] += a

    dx[1:9, 4:7] = paccelerations
    asteroid_accelerations = get_paccelerations(None, input[9:, 0:4], amodel)
    asteroid_accelerations = asteroid_accelerations.reshape(nbodies - 9, 4)
    asteroid_accelerations = asteroid_accelerations[:, 1:].reshape(-1).reshape((nbodies - 9, 3))
    asteroid_accelerations = asteroid_accelerations * G * 1e-12
    for i in range(9, nbodies):
        for j in range(0, 9):
            r = input[i, 1:4] - input[j, 1:4]
            r = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
            a = -G * input[j, 0] * (input[i, 1:4] - input[j, 1:4]) / r ** 3
            asteroid_accelerations[i - 9, :] += a
    dx[9:, 4:7] = asteroid_accelerations
    return dx


def get_real(input):
    dx = np.zeros_like(input)
    for i in range(0, nbodies):
        dx[i][0] = 0
    dx[:, 1:4] = input[:, 4:7]
    dx[0:, 4:7] = get_accelerations(input) * G
    dx[0, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    return dx


valid_asteroids = [2, 3, 4, 5, 10, 20, 30, 40, 50]

for a in valid_asteroids:
    nbodies = 9 + a
    np.random.seed(int(time.time()))
    t_points = 1000
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    planets = solarsys()
    state = np.zeros((nbodies, 7))
    state[1:9, 0:4] = planets
    state[0, :] = [1, 0, 0, 0, 0, 0, 0]
    state[1, 4:7] = velocity(8, 13, state[1, 1:4])
    state[2, 4:7] = velocity(7, 8, state[2, 1:4])
    state[3, 4:7] = velocity(6, 7, state[3, 1:4])
    state[4, 4:7] = velocity(4, 6, state[4, 1:4])
    state[5, 4:7] = velocity(2, 3, state[5, 1:4])
    state[6, 4:7] = velocity(1, 2, state[6, 1:4])
    state[7, 4:7] = velocity(1, 2, state[7, 1:4])
    state[8, 4:7] = velocity(1, 1.5, state[8, 1:4])
    state[9:, 0:4] = asteroids(a)
    for j in range(9, nbodies):
        state[j, 4:7] = velocity(2, 5, state[j, 1:4])
        state[j, 0] = state[j, 0] * 1e-12

    G = 39.47841760435743

    planet_args = ObjectView(get_args(True, 0))
    asteroid_args = ObjectView(get_args(False, a))
    porbit = np.zeros((nbodies, 7, t_points))
    porbit[:, :, 0] = state
    planet_model = load_model(planet_args)
    asteroid_model = load_model(asteroid_args)
    prev = np.zeros(3)
    starttime = time.time()
    flags = 0
    for i in range(1, t_points):
        # dx = get_next(None, porbit[:, :, i - 1], planet_model, asteroid_model)
        # # dx = get_real(porbit[:, :, i - 1])
        # porbit[:, :, i] = porbit[:, :, i - 1] + (dx * (t_eval[i] - t_eval[i - 1]))

        # use rk4
        h = t_eval[i] - t_eval[i - 1]
        k1 = get_next(None, porbit[:, :, i - 1], planet_model, asteroid_model)
        k2 = get_next(None, porbit[:, :, i - 1] + (h / 2) * k1, planet_model, asteroid_model)
        k3 = get_next(None, porbit[:, :, i - 1] + (h / 2) * k2, planet_model, asteroid_model)
        k4 = get_next(None, porbit[:, :, i - 1] + h * k3, planet_model, asteroid_model)
        acc = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h
        # sum of all acc together
        next = np.zeros(3)
        for j in range(0, nbodies):
            next[0] += acc[j][1]
            next[1] += acc[j][2]
            next[2] += acc[j][3]

        R = abs((((prev[0] - next[0]) ** 2 + (prev[1] - next[1]) ** 2 + (prev[2] - next[2]) ** 2) ** 0.5) / (
                ((prev[0] ** 2 + prev[1] ** 2 + prev[2] ** 2) ** 0.5) + 1e-11))
        if (R > 0):
            k1 = get_real(porbit[:, :, i - 1])
            k2 = get_real(porbit[:, :, i - 1] + (h / 2) * k1)
            k3 = get_real(porbit[:, :, i - 1] + (h / 2) * k2)
            k4 = get_real(porbit[:, :, i - 1] + h * k3)
            acc = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h
            next = np.zeros(3)
            for j in range(0, nbodies):
                next[0] = acc[j][1]
                next[1] = acc[j][2]
                next[2] = acc[j][3]
            flags += 1
        prev = next
        porbit[:, :, i] = porbit[:, :, i - 1] + acc

    endtime = time.time()
    print('hnn')
    print('asteroids: {}'.format(a))
    print('time taken: {}'.format(endtime - starttime))
    print('flags: {}'.format(flags))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    for i in range(9, nbodies):
        names.append('Asteroid {}'.format(i - 8))
    for i in range(0, nbodies):
        ax.plot(porbit[i, 1, :], porbit[i, 2, :], porbit[i, 3, :], label=names[i])
    ax.legend()
    plt.title('hnn {}'.format(i))
    plt.show()

    rorbit = np.zeros((nbodies, 7, t_points))
    rorbit[:, :, 0] = state

    starttime = time.time()
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    for i in range(0, nbodies):
        sim.add(m=state[i][0], x=state[i][1], y=state[i][2], z=state[i][3], vx=state[i][4], vy=state[i][5],
                vz=state[i][6])

    sim.integrator = 'ias15'
    sim.dt = 0.0001
    sim.move_to_com()

    for i in range(1, t_points):
        sim.integrate(t_eval[i])
        for j in range(0, nbodies):
            rorbit[j, :, i] = [sim.t, sim.particles[j].x, sim.particles[j].y, sim.particles[j].z,
                               sim.particles[j].vx, sim.particles[j].vy, sim.particles[j].vz]

    endtime = time.time()
    print('real')
    print('asteroids: {}'.format(a))
    print('time taken: {}'.format(endtime - starttime))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, nbodies):
        ax.plot(rorbit[i, 1, :], rorbit[i, 2, :], rorbit[i, 3, :], label=names[i])
    ax.legend()
    plt.title('real {}'.format(i))
    plt.show()
