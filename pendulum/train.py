import argparse
import os
import sys

import numpy as np
import torch

from data import get_dataset
from MLP import MLP
from nn import nn_model

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


# util functions

def rk4(fun, y0, t, dt, *args, **kwargs):
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0 + dt / 2.0 * k1, t + dt / 2.0, *args, **kwargs)
    k3 = fun(y0 + dt / 2.0 * k2, t + dt / 2.0, *args, **kwargs)
    k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)

    dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return dy


def loss(u, v):
    return (u - v).pow(2).mean()


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print("Training baseline neural network model:")

    output_dim = args.input_dim
    mlp = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = nn_model(args.input_dim, differentiable_model=mlp, field_type=args.field_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=1e-4)

    data = get_dataset(seed=args.seed)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dx = torch.Tensor(data['dx'])
    test_dx = torch.Tensor(data['test_dx'])

    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        dx_hat = model.time_deriv(x)
        l2_loss = loss(dx_hat, dx)
        l2_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        test_dx_hat = model.time_deriv(test_x)
        test_l2_loss = loss(test_dx_hat, test_dx)

        stats['train_loss'].append(l2_loss.item())
        stats['test_loss'].append(test_l2_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, l2_loss.item(), test_l2_loss.item()))

    train_dx_hat = model.time_deriv(x)
    train_dist = (dx - train_dx_hat).pow(2)
    test_dx_hat = model.time_deriv(test_x)
    test_dist = (test_dx - test_dx_hat).pow(2)
    print("Final train loss {:.4e} +/- {:.4e}".format(train_dist.mean().item(), train_dist.std().item()))
    return model, stats


if __name__ == '__main__':
    args = get_args()
    model, stats = train(args)

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    path = '{}/{}.pt'.format(args.save_dir, args.name)
    torch.save(model.state_dict(), path)
