import numpy as np
from data import kep2cart, gen_orbital_elements, update
from utils import from_pickle, to_pickle, mse_loss


def asteroids(n):
    state = np.zeros((n, 4))
    for b in range(0, n):
        state[b][0] = np.random.uniform(1, 100)
        a, e, i, true, long, argp = gen_orbital_elements([2, 4], [0.0, 0.1], [0, 8], [0, 360], [0, 0], [0, 0])
        state[b][1:] = kep2cart(a, e, i, long, argp, true)
    return state


def sample_asteroids(nbodies, timesteps=60, samples=50000, t_span=[0, 20], verbose=True, **kwargs):
    orbit_settings = locals()
    if verbose:
        print('Sampling orbits...')

    x = []
    dx = []
    # e = []
    N = timesteps * samples
    while len(x) < N:
        state = asteroids(nbodies)

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


def make_dataset(name, num_asteroids, test_split=0.15, **kwargs):
    data, settings = sample_asteroids(num_asteroids, **kwargs)
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data
    data['meta'] = settings
    return data, settings


def get_dataset(name, num_asteroids, save_dir, **kwargs):
    path = '{}/{}-{}.pkl'.format(save_dir, num_asteroids, name)
    try:
        data = from_pickle(path)
        print("Loaded dataset from {}".format(path))
    except:
        data, settings = make_dataset(name, num_asteroids, **kwargs)
        to_pickle(data, path)
        print("Saved dataset to {}".format(path))

    return data


import torch
from nn_models import MLP
from hnn import hnn
import argparse
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='nonlinearity')
    parser.add_argument('--hidden_dim', default=300, type=int, help='hidden dimension')
    parser.add_argument('--input_dim', default=6, type=int, help='input dimension')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=20000, type=int, help='batch size')
    parser.add_argument('--total_steps', default=2000, type=int, help='total number of gradient steps')
    parser.add_argument('--print_every', default=1000, type=int, help='print every')
    parser.add_argument('--name', default='asteroids', type=str, help='save name')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='field type')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--num_asteroids', default=5, type=int, help='asteroids number')
    parser.add_argument('--save_dir', default='/storage/models', type=str, help='save dir')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = hnn(args.input_dim, differentiable_model=nn_model, field_type=args.field_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    data = get_dataset(args.name, args.num_asteroids, args.save_dir)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)
    dx = torch.Tensor(data['dcoords'])
    test_dx = torch.Tensor(data['test_dcoords'])

    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dx_hat = model.time_deriv(x[ixs])
        loss = mse_loss(dx[ixs], dx_hat)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optimizer.step()
        optimizer.zero_grad()

        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dx_hat = model.time_deriv(test_x[test_ixs])
        test_loss = mse_loss(test_dx[test_ixs], test_dx_hat)

        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())

        if args.verbose and step % args.print_every == 0:
            print(
                "step {}, loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std{:.4e}".format(step, loss, test_loss,
                                                                                                  grad @ grad,
                                                                                                  grad.std()))

    train_dx_hat = model.time_deriv(x)
    train_dist = (dx - train_dx_hat)
    test_dx_hat = model.time_deriv(test_x)
    test_dist = (test_dx - test_dx_hat)
    print('Final train loss {:.4e} +/- {:.4e}'.format(train_dist.mean().item(),
                                                      train_dist.std().item() / np.sqrt(train_dist.shape[0])))
    print('Final test loss {:.4e} +/- {:.4e}'.format(test_dist.mean().item(),
                                                     test_dist.std().item() / np.sqrt(test_dist.shape[0])))
    return model, stats


if __name__ == '__main__':
    print("Training HNN")
    args = get_args()
    args.verbose = True
    copy = args.num_asteroids
    valid_asteroids = [50, 60, 70, 80, 90]
    for i in valid_asteroids:
        args.num_asteroids = int(i)
        args.input_dim = 3 * i
        model, stats = train(args)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        label = 'hnn'
        model_path = '{}/{}{}_{}.pt'.format(args.save_dir, i, args.name, label)
        torch.save(model.state_dict(), model_path)

        stats_path = '{}/{}{}_{}.pkl'.format(args.save_dir, i, args.name, label)
        to_pickle(stats, stats_path)
