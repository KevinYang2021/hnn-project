import matplotlib.pyplot as plt
import torch, time, os, sys
import numpy as np
import scipy.integrate, scipy.ndimage
from nn_models import MLP
from hnn import hnn
from data import get_dataset, solarsys, potential_energy, kinetic_energy, total_energy, get_accelerations
import rebound


def get_args():
    return {
        'nonlinearity': 'tanh', 'hidden_dim': 300, 'input_dim': 24, 'learning_rate': 0.001, 'batch_size': 20000,
        'total_steps': 2000, 'print_every': 1000, 'name': 'solarsys', 'verbose': True, 'field_type': 'solenoidal',
        'seed': 0, 'save_dir': '/home/kevin/code/project/final'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def velocity(min, max, pos):
    x, y, z = pos
    vel = np.random.uniform(min, max)
    vx = -y * vel / np.sqrt(x ** 2 + y ** 2)
    vy = x * vel / np.sqrt(x ** 2 + y ** 2)
    vz = 0
    return np.array([vx, vy, vz])


args = ObjectView(get_args())
nbodies = (int)(args.input_dim / 3) + 1

np.random.seed(int(time.time()))
t_points = 10000
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], t_points)
planets = solarsys()
state = np.zeros((nbodies, 7))
state[1:, 0:4] = planets
state[0, :] = [1, 0, 0, 0, 0, 0, 0]
state[1, 4:7] = velocity(8, 13, state[1, 1:4])
state[2, 4:7] = velocity(7, 8, state[2, 1:4])
state[3, 4:7] = velocity(6, 7, state[3, 1:4])
state[4, 4:7] = velocity(4, 6, state[4, 1:4])
state[5, 4:7] = velocity(2, 3, state[5, 1:4])
state[6, 4:7] = velocity(1, 2, state[6, 1:4])
state[7, 4:7] = velocity(1, 2, state[7, 1:4])
state[8, 4:7] = velocity(1, 1.5, state[8, 1:4])

G = 39.47841760435743


# G = 1


def load_model(args):
    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = hnn(args.input_dim, differentiable_model=nn_model, field_type=args.field_type)
    label = 'hnn'
    path = '{}/{}_{}.pt'.format(args.save_dir, args.name, label)
    model.load_state_dict(torch.load(path))
    return model


hnn_model = load_model(args)


def get_paccelerations(t, input, model):
    input = input.reshape(-1, 4)
    deriv = np.zeros_like(input)
    np_x = input[:, 1:]  # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_deriv(x)
    deriv[:, 1:] = dx_hat.detach().data.numpy().reshape(3, nbodies - 1).T
    return deriv.reshape(-1)


def get_next(t, input, model):
    paccelerations = get_paccelerations(None, input[1:, 0:4], model)
    paccelerations = paccelerations.reshape(nbodies - 1, 4)
    paccelerations = paccelerations[:, 1:].reshape(-1).reshape((nbodies - 1, 3))
    paccelerations = paccelerations * G
    dx = np.zeros_like(input)
    for i in range(0, nbodies):
        dx[i][0] = 0
    dx[:, 1:4] = input[:, 4:7]
    dx[0, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    for i in range(1, nbodies):
        r = input[i, 1:4] - input[0, 1:4]
        r = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
        a = -G * input[0, 0] * (input[i, 1:4] - input[0, 1:4]) / r ** 3
        paccelerations[i - 1, :] += a

    dx[1:, 4:7] = paccelerations
    return dx


def get_real(input):
    dx = np.zeros_like(input)
    for i in range(0, nbodies):
        dx[i][0] = 0
    dx[:, 1:4] = input[:, 4:7]
    dx[0:, 4:7] = get_accelerations(input) * G
    dx[0, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    return dx


porbit = np.zeros((nbodies, 7, t_points))
porbit[:, :, 0] = state

for i in range(1, t_points):
    # dx = get_next(None, porbit[:, :, i - 1], hnn_model)
    # dx = get_real(porbit[:, :, i - 1])
    # porbit[:, :, i] = porbit[:, :, i - 1] + (dx * (t_eval[i] - t_eval[i - 1]))

    # use rk4
    h = t_eval[i] - t_eval[i - 1]
    k1 = get_next(None, porbit[:, :, i - 1], hnn_model)
    k2 = get_next(None, porbit[:, :, i - 1] + (h / 2) * k1, hnn_model)
    k3 = get_next(None, porbit[:, :, i - 1] + (h / 2) * k2, hnn_model)
    k4 = get_next(None, porbit[:, :, i - 1] + h * k3, hnn_model)
    porbit[:, :, i] = porbit[:, :, i - 1] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for i in range(0, nbodies):
    ax.plot(porbit[i, 1, :], porbit[i, 2, :], porbit[i, 3, :], label=names[i])
ax.legend()
plt.show()

rorbit = np.zeros((nbodies, 7, t_points))
rorbit[:, :, 0] = state

sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
for i in range(0, nbodies):
    sim.add(m=state[i][0], x=state[i][1], y=state[i][2], z=state[i][3], vx=state[i][4], vy=state[i][5], vz=state[i][6])

sim.integrator = 'whfast'
sim.dt = 0.001
sim.move_to_com()

for i in range(1, t_points):
    sim.integrate(t_eval[i])
    for j in range(0, nbodies):
        rorbit[j, :, i] = [sim.t, sim.particles[j].x, sim.particles[j].y, sim.particles[j].z,
                           sim.particles[j].vx, sim.particles[j].vy, sim.particles[j].vz]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, nbodies):
    ax.plot(rorbit[i, 1, :], rorbit[i, 2, :], rorbit[i, 3, :], label=names[i])
ax.legend()
plt.show()
