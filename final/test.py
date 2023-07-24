import matplotlib.pyplot as plt
import torch, time, os, sys
import numpy as np
import scipy.integrate, scipy.ndimage
from nn_models import MLP
from hnn import hnn
from data import get_dataset, sjs, potential_energy, kinetic_energy, total_energy, get_accelerations
import rebound


def get_args():
    return {
        'nonlinearity': 'tanh', 'hidden_dim': 300, 'input_dim': 6, 'learning_rate': 0.001, 'batch_size': 20000,
        'total_steps': 2000, 'print_every': 1000, 'name': 'sjs', 'verbose': True, 'field_type': 'solenoidal',
        'seed': 0, 'save_dir': '/home/kevin/code/project/final'}


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


args = ObjectView(get_args())
nbodies = (int)(args.input_dim / 3) + 1

np.random.seed(int(time.time()))
t_points = 10000
t_span = [0, 500]
t_eval = np.linspace(t_span[0], t_span[1], t_points)
planets = sjs()
state = np.zeros((nbodies, 7))
state[1:, 0:4] = planets
state[0, :] = [1, 0, 0, 0, 0, 0, 0]
jx, jy, jz = state[1, 1:4]
sx, sy, sz = state[2, 1:4]
jvel = np.random.uniform(2.62421, 2.89423)
svel = np.random.uniform(1.92808, 2.13903)
jvx = -jy * jvel / np.sqrt(jx ** 2 + jy ** 2)
jvy = jx * jvel / np.sqrt(jx ** 2 + jy ** 2)
jvz = 0
svx = -sy * svel / np.sqrt(sx ** 2 + sy ** 2)
svy = sx * svel / np.sqrt(sx ** 2 + sy ** 2)
svz = 0
state[1][4] = jvx
state[1][5] = jvy
state[1][6] = jvz
state[2][4] = svx
state[2][5] = svy
state[2][6] = svz

real_orbit = np.zeros((nbodies, 7, t_points))
real_orbit[:, :, 0] = state

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


def get_paccelerations(t, state, model):
    state = state.reshape(-1, 4)
    deriv = np.zeros_like(state)
    np_x = state[:, 1:]  # drop mass
    np_x = np_x.T.flatten()[None, :]
    x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
    dx_hat = model.time_deriv(x)
    deriv[:, 1:] = dx_hat.detach().data.numpy().reshape(3, nbodies - 1).T
    return deriv.reshape(-1)


hnn_orbit = np.zeros((nbodies, 7, t_points))
hnn_orbit[:, :, 0] = state

for i in range(1, t_points):
    dx = get_paccelerations(None, hnn_orbit[1:, 0:4, i - 1], hnn_model)
    dx = dx.reshape(2, 4)
    dx = dx[:, 1:].reshape(-1).reshape((2, 3))
    G = 39.47841760435743
    r = hnn_orbit[1, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]
    r = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5
    a = -G * hnn_orbit[0, 0, i - 1] * (hnn_orbit[1, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]) / (r ** 3)
    rr = hnn_orbit[2, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]
    rr = (rr[0] ** 2 + rr[1] ** 2 + rr[2] ** 2) ** 0.5
    aa = -G * hnn_orbit[0, 0, i - 1] * (hnn_orbit[2, 1:4, i - 1] - hnn_orbit[0, 1:4, i - 1]) / (rr ** 3)
    dx += (np.array([[a[0], a[1], a[2]], [aa[0], aa[1], aa[2]]]) * (t_eval[i] - t_eval[i - 1]))
    hnn_orbit[0, :, i] = hnn_orbit[0, :, i - 1]
    hnn_orbit[1:, 0, i] = hnn_orbit[1:, 0, i - 1]
    hnn_orbit[1:, 4:7, i] = hnn_orbit[1:, 4:7, i - 1] + dx
    hnn_orbit[1:, 1:4, i] = hnn_orbit[1:, 4:7, i] * (t_eval[i] - t_eval[i - 1]) + hnn_orbit[1:, 1:4, i - 1]

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

# graph energies
rtot = total_energy(real_orbit)
htot = total_energy(hnn_orbit)
rkin = kinetic_energy(real_orbit)
hkin = kinetic_energy(hnn_orbit)
rpot = potential_energy(real_orbit)
hpot = potential_energy(hnn_orbit)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(t_eval, rtot, label='Real')
ax.plot(t_eval, htot, label='HNN')
plt.title('Total Energy')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(t_eval, rkin, label='Real')
ax.plot(t_eval, hkin, label='HNN')
plt.title('Kinetic Energy')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(t_eval, rpot, label='Real')
ax.plot(t_eval, hpot, label='HNN')
plt.title('Potential Energy')
plt.legend()
plt.show()

cnt = 10000
# create 2x3 subplot config
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].set_xlabel('Real Jupiter X Acceleration')
axs[0, 0].set_ylabel('Predicted Jupiter X Acceleration')
axs[0, 0].set_title('Jupiter X Acceleration')
axs[0, 1].set_xlabel('Real Jupiter Y Acceleration')
axs[0, 1].set_ylabel('Predicted Jupiter Y Acceleration')
axs[0, 1].set_title('Jupiter Y Acceleration')
axs[0, 2].set_xlabel('Real Jupiter Z Acceleration')
axs[0, 2].set_ylabel('Predicted Jupiter Z Acceleration')
axs[0, 2].set_title('Jupiter Z Acceleration')
axs[1, 0].set_xlabel('Real Saturn X Acceleration')
axs[1, 0].set_ylabel('Predicted Saturn X Acceleration')
axs[1, 0].set_title('Saturn X Acceleration')
axs[1, 1].set_xlabel('Real Saturn Y Acceleration')
axs[1, 1].set_ylabel('Predicted Saturn Y Acceleration')
axs[1, 1].set_title('Saturn Y Acceleration')
axs[1, 2].set_xlabel('Real Saturn Z Acceleration')
axs[1, 2].set_ylabel('Predicted Saturn Z Acceleration')
axs[1, 2].set_title('Saturn Z Acceleration')

juxa = []
juxp = []
juya = []
juyp = []
juza = []
juzp = []
suxa = []
suxp = []
suya = []
suyp = []
suza = []
suzp = []

for i in range(cnt):
    planets = sjs()
    state = np.zeros((nbodies, 7))
    state[1:, 0:4] = planets
    state[0, :] = [1, 0, 0, 0, 0, 0, 0]
    jx, jy, jz = state[1, 1:4]
    sx, sy, sz = state[2, 1:4]
    jvel = np.random.uniform(2.62421, 2.89423)
    svel = np.random.uniform(1.92808, 2.13903)
    jvx = -jy * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvy = jx * jvel / np.sqrt(jx ** 2 + jy ** 2)
    jvz = 0
    svx = -sy * svel / np.sqrt(sx ** 2 + sy ** 2)
    svy = sx * svel / np.sqrt(sx ** 2 + sy ** 2)
    svz = 0
    state[1][4] = jvx
    state[1][5] = jvy
    state[1][6] = jvz
    state[2][4] = svx
    state[2][5] = svy
    state[2][6] = svz

    acc = get_accelerations(state)
    hnn_acc = get_paccelerations(None, planets, hnn_model)
    # G = 39.47841760435743
    G = 1
    r = np.linalg.norm(state[1, 1:4] - state[0, 1:4])
    a = -G * state[0, 0] * (state[1, 1:4] - state[0, 1:4]) / r ** 3
    rr = np.linalg.norm(state[2, 1:4] - state[0, 1:4])
    aa = -G * state[0, 0] * (state[2, 1:4] - state[0, 1:4]) / rr ** 3
    # hnn_acc = np.array([0, a[0], a[1], a[2], 0, aa[0], aa[1], aa[2]])
    acc[1:, :] -= ([[a[0], a[1], a[2]], [aa[0], aa[1], aa[2]]])

    acc = acc[1:, :]
    acc = acc.flatten()
    hnn_acc = hnn_acc.reshape((2, 4))
    hnn_acc = hnn_acc[:, 1:]
    hnn_acc = hnn_acc.flatten()

    juxa.append(acc[0])
    juxp.append(hnn_acc[0])
    juya.append(acc[1])
    juyp.append(hnn_acc[1])
    juza.append(acc[2])
    juzp.append(hnn_acc[2])
    suxa.append(acc[3])
    suxp.append(hnn_acc[3])
    suya.append(acc[4])
    suyp.append(hnn_acc[4])
    suza.append(acc[5])
    suzp.append(hnn_acc[5])

axs[0, 0].scatter(juxa, juxp, color='blue')
axs[0, 1].scatter(juya, juyp, color='blue')
axs[0, 2].scatter(juza, juzp, color='blue')
axs[1, 0].scatter(suxa, suxp, color='blue')
axs[1, 1].scatter(suya, suyp, color='blue')
axs[1, 2].scatter(suza, suzp, color='blue')

for i in range(2):
    for j in range(3):
        axs[i, j].axline((0, 0), slope=1, color='red')

plt.tight_layout()
plt.show()
