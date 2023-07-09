import matplotlib.pyplot as plt
import torch, time, os, sys
import numpy as np
import scipy.integrate, scipy.ndimage
from nn_models import MLP
from hnn import hnn
from data import get_dataset, sjs, potential_energy, kinetic_energy, total_energy
import rebound


def get_args():
    return {
        'nonlinearity': 'tanh', 'hidden_dim': 300, 'input_dim': 12, 'learning_rate': 0.001, 'batch_size': 20000,
        'total_steps': 10000, 'print_every': 1000, 'name': 'sjs', 'verbose': True, 'field_type': 'solenoidal',
        'seed': 0, 'save_dir': '/home/kevin/code/project/final'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


args = ObjectView(get_args())

t_points = 10000
t_span = [0, 5000]
t_eval = np.linspace(t_span[0], t_span[1], t_points)
state = sjs()

nbodies = args.input_dim // 6 + 1
real_orbit = np.zeros((nbodies, 7, t_points))
real_orbit[0, :, 0] = [1, 0, 0, 0, 0, 0, 0]
real_orbit[1:, :, 0] = state

sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
sim.add(m=1, x=0, y=0, z=0, vx=0, vy=0, vz=0)
for i in range(0, nbodies - 1):
    sim.add(m=state[i][0], x=state[i][1], y=state[i][2], z=state[i][3], vx=state[i][4], vy=state[i][5], vz=state[i][6])
sim.integrator = 'whfast'
sim.dt = 0.01
sim.move_to_com()

for i in range(1, t_points):
    sim.integrate(t_eval[i])
    for j in range(0, nbodies):
        real_orbit[j, :, i] = [sim.t, sim.particles[j].x, sim.particles[j].y, sim.particles[j].z,
                               sim.particles[j].vx, sim.particles[j].vy, sim.particles[j].vz]

# graph real orbit
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, nbodies):
    ax.plot(real_orbit[i, 1, :], real_orbit[i, 2, :], real_orbit[i, 3, :])
plt.show()


def load_model(args):
    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = hnn(args.input_dim, differentiable_model=nn_model, field_type=args.field_type)
    label = 'hnn'
    path = '{}/{}_{}.pt'.format(args.save_dir, args.name, label)
    model.load_state_dict(torch.load(path))
    return model


hnn_model = load_model(args)


def model_update(t, state, model):
    state = state.reshape(-1, 7)
    deriv = np.zeros_like(state)
    np_x = state[:, 1:]  # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_deriv(x)
    deriv[:, 1:] = dx_hat.detach().data.numpy().reshape(6, nbodies - 1).T
    return deriv.reshape(-1)


hnn_orbit = np.zeros((nbodies, 7, t_points))
hnn_orbit[0, :, 0] = [1, 0, 0, 0, 0, 0, 0]
hnn_orbit[1:, :, 0] = state

for i in range(1, t_points):
    dx = model_update(t_eval[i - 1], hnn_orbit[1:, :, i - 1], hnn_model) * (t_eval[i] - t_eval[i - 1])
    G = 39.476926421373
    r = np.linalg.norm(hnn_orbit[1, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1])
    a = -G * (hnn_orbit[1, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]) / r ** 3 * (t_eval[i] - t_eval[i - 1])
    rr = np.linalg.norm(hnn_orbit[2, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1])
    aa = -G * (hnn_orbit[2, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]) / rr ** 3 * (t_eval[i] - t_eval[i - 1])
    dx[1:4] = hnn_orbit[1, 4:7, i - 1] * (t_eval[i] - t_eval[i - 1])
    dx[4:7] += a
    dx[8:11] = hnn_orbit[2, 4:7, i - 1] * (t_eval[i] - t_eval[i - 1])
    dx[11:14] += aa
    deriv = np.zeros((nbodies, 7))
    deriv[1:, :] = dx.reshape(nbodies - 1, 7)
    hnn_orbit[:, :, i] = hnn_orbit[:, :, i - 1] + deriv

# graph hnn orbit
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, nbodies):
    ax.plot(hnn_orbit[i, 1, :], hnn_orbit[i, 2, :], hnn_orbit[i, 3, :])
plt.show()

# graph real and hnn orbit
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, nbodies):
    ax.plot(real_orbit[i, 1, :], real_orbit[i, 2, :], real_orbit[i, 3, :])
for i in range(0, nbodies):
    ax.plot(hnn_orbit[i, 1, :], hnn_orbit[i, 2, :], hnn_orbit[i, 3, :])
# give title
plt.title('Real and HNN Orbit')
plt.show()

print(hnn_orbit[1, :, 0])
print(hnn_orbit[2, :, 0])
dx = model_update(t_eval[0], hnn_orbit[1:, :, 0], hnn_model) * (t_eval[1] - t_eval[0])
G = 39.476926421373
r = np.linalg.norm(hnn_orbit[1, 1:4, 0] - hnn_orbit[0, 1:4, 0])
a = -G * (hnn_orbit[1, 1:4, 0] - hnn_orbit[0, 1:4, 0]) / r ** 3 * (t_eval[1] - t_eval[0])
rr = np.linalg.norm(hnn_orbit[2, 1:4, 0] - hnn_orbit[0, 1:4, 0])
aa = -G * (hnn_orbit[2, 1:4, 0] - hnn_orbit[0, 1:4, 0]) / rr ** 3 * (t_eval[1] - t_eval[0])
dx[1:4] = hnn_orbit[1, 4:7, 0] * (t_eval[1] - t_eval[0])
dx[4:7] += a
dx[8:11] = hnn_orbit[2, 4:7, 0] * (t_eval[1] - t_eval[0])
dx[11:14] += aa
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
sim.add(m=1, x=0, y=0, z=0, vx=0, vy=0, vz=0)
for i in range(0, nbodies - 1):
    sim.add(m=state[i][0], x=state[i][1], y=state[i][2], z=state[i][3], vx=state[i][4], vy=state[i][5], vz=state[i][6])
sim.integrator = 'whfast'
sim.dt = 0.01
sim.move_to_com()
sim.integrate(t_eval[1])
rdx = np.array(
    [0, sim.particles[1].x, sim.particles[1].y, sim.particles[1].z, sim.particles[1].vx, sim.particles[1].vy,
     sim.particles[1].vz, 0, sim.particles[2].x, sim.particles[2].y, sim.particles[2].z,
     sim.particles[2].vx, sim.particles[2].vy, sim.particles[2].vz])
rdx -= np.array([0, state[0][1], state[0][2], state[0][3], state[0][4], state[0][5], state[0][6], 0, state[1][1],
                 state[1][2], state[1][3], state[1][4], state[1][5], state[1][6]])
print(dx)
print(rdx)

print(dx - rdx)
