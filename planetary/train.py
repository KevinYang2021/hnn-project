import torch, argparse
import numpy as np
import os, sys, pickle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PARENT_DIR)

from nn_model import MLP
from hnn import hnn
from data import get_dataset


def from_pickle(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def to_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def l1_loss(u, v):
    return torch.mean((u - v))


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='nonlinearity')
    parser.add_argument('--hidden_dim', default=300, type=int, help='hidden dimension')
    parser.add_argument('--input_dim', default=18, type=int, help='input dimension')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=25000, type=int, help='batch size')
    parser.add_argument('--total_steps', default=20000, type=int, help='total number of gradient steps')
    parser.add_argument('--print_every', default=1000, type=int, help='print every')
    parser.add_argument('--name', default='sjs', type=str, help='save name')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='field type')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='save dir')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print("Training HNN")

    output_dim = 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = hnn(args.input_dim, differentiable_model=nn_model, field_type=args.field_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    data = get_dataset(args.name, args.save_dir)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)
    dx = torch.Tensor(data['dcoords'])
    test_dx = torch.Tensor(data['test_dcoords'])

    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dx_hat = model.time_deriv(x[ixs])
        loss = l1_loss(dx[ixs], dx_hat)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optimizer.step()
        optimizer.zero_grad()

        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dx_hat = model.time_deriv(test_x[test_ixs])
        test_loss = l1_loss(test_dx[test_ixs], test_dx_hat)

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
    model, stats = train(args)

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'hnn'
    model_path = '{}/{}_{}.pt'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), model_path)

    stats_path = '{}/{}_{}.pkl'.format(args.save_dir, args.name, label)
    to_pickle(stats, stats_path)
