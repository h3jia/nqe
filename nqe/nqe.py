import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from .interp import Interp1D

__all__ = ['QuantileLoss', 'MLP', 'QuantileMLP1D', 'NQE']


class QuantileLoss:

    def __init__(self, quantiles, alpha=0.):
        self.quantiles = torch.as_tensor(quantiles, dtype=torch.float)
        self.alpha = float(alpha)
        self._weights = torch.exp(self.alpha * np.abs(self.quantiles - 0.5))[None]

    def __call__(self, input, target):
        # in_now shape: # of points, (# of data dims + # of previous theta dims)
        # input = model(in_now) shape: # of points, # of quantiles
        # target = out_now shape: # of points
        if target.ndim == input.ndim:
            pass
        elif target.ndim == input.ndim - 1:
            target = target[..., None]
        else:
            raise RuntimeError
        weights = torch.where(target > input, self.quantiles, 1 - self.quantiles)
        return torch.mean(torch.abs(weights * self._weights * (input - target)))


class MLP(nn.Module):

    def __init__(self, in_features, out_features, add_fc, nonlinear='relu', batch_norm=False,
                 shortcut=True):
        super(MLP, self).__init__()
        self._make_fc(in_features, out_features, add_fc, nonlinear, batch_norm, shortcut)
        self.mu = 0.
        self.sigma = 1.

    def _make_fc(self, in_features: int, out_features: int, add_fc: Optional[List[int]],
                 nonlinear: str = 'tanh', batch_norm: bool = True, shortcut: bool = True):
        if nonlinear.lower() == 'tanh':
            self.nonlinear = nn.Tanh()
        elif nonlinear.lower() == 'relu':
            self.nonlinear = nn.ReLU()
        elif nonlinear.lower() == 'leakyrelu':
            self.nonlinear = nn.LeakyReLU()
        elif nonlinear.lower() == 'elu':
            self.nonlinear = nn.ELU()
        else:
            raise NotImplementedError
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.batch_norm = bool(batch_norm)
        self.shortcut = bool(shortcut)
        if add_fc is None:
            self.fc_layers.append(nn.Linear(in_features, out_features))
        else:
            add_fc = list(add_fc)
            add_fc.insert(0, in_features)
            add_fc.append(out_features)
            self.fc_layers.append(nn.Linear(add_fc[0], add_fc[1]))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(add_fc[1]))
            # pass thru nonlinear, concat with shortcut
            k = add_fc[0] if shortcut else 0
            for i in range(1, len(add_fc) - 2):
                self.fc_layers.append(nn.Linear(add_fc[i] + k, add_fc[i + 1]))
                if batch_norm:
                    self.bn_layers.append(nn.BatchNorm1d(add_fc[i + 1]))
                # pass thru nonlinear, concat with shortcut
            self.fc_layers.append(nn.Linear(add_fc[-2] + k, add_fc[-1]))

    def set_whitening(self, x):
        self.mu = torch.mean(x, axis=0).detach()
        self.sigma = torch.std(x, axis=0)
        self.sigma = torch.where(self.sigma > 0., self.sigma, 1e-12).detach()

    def _forward(self, x):
        x = (x - self.mu) / self.sigma
        if len(self.fc_layers) == 1:
            return self.fc_layers[0](x)
        elif len(self.fc_layers) > 1:
            if self.shortcut:
                x_clone = torch.clone(x)
            x = self.fc_layers[0](x)
            if self.batch_norm:
                x = self.bn_layers[0](x)
            x = self.nonlinear(x)
            for i in range(1, len(self.fc_layers) - 1):
                if self.shortcut:
                    x = torch.concat((x, x_clone), axis=-1)
                x = self.fc_layers[i](x)
                if self.batch_norm:
                    x = self.bn_layers[i](x)
                x = self.nonlinear(x)
            if self.shortcut:
                x = torch.concat((x, x_clone), axis=-1)
            return self.fc_layers[-1](x)
        else:
            raise RuntimeError

    forward = _forward

    __call__ = _forward


class QuantileMLP1D(MLP):

    def __init__(self, low, high, quantile_method='cumsum', binary_depth=0, *args, **kwargs):
        super(QuantileMLP1D, self).__init__(*args, **kwargs)
        self.low = float(low)
        self.high = float(high)
        self.quantile_method = str(quantile_method)
        assert self.quantile_method in ('cumsum', 'binary')
        self.binary_depth = int(binary_depth)

    def forward(self, x, return_raw=False):
        x = self._forward(x)
        if self.quantile_method == 'cumsum':
            y = self.low + (self.high - self.low) * torch.cumsum(torch.softmax(x, axis=-1),
                                                                 axis=-1)[..., :-1]
            return (y, x) if return_raw else y
        elif self.quantile_method == 'binary':
            if return_raw:
                raise NotImplementedError
            x = x.contiguous()
            assert x.ndim == 2
            x = x.reshape((x.shape[0], -1, 2))
            x = torch.softmax(x, axis=-1)
            y = torch.repeat_interleave(x[:, 0, :], 2**(self.binary_depth - 1), -1)
            offset = 1
            for i in range(1, self.binary_depth):
                for j in range(0, 2**i):
                    k0 = j * 2**(self.binary_depth - i)
                    k1 = (j + 1) * 2**(self.binary_depth - i)
                    y[..., k0:k1] *= torch.repeat_interleave(x[:, offset, :],
                                                             2**(self.binary_depth - i - 1), -1)
                    offset += 1
            return self.low + (self.high - self.low) * torch.cumsum(y, axis=-1)[..., :-1]
        else:
            return ValueError

    __call__ = forward


class NQE:

    def __init__(self, quantiles, lows, highs, quantile_method='cumsum', alpha=0.,
                 add_fc=(256,) * 6, nonlinear='relu', batch_norm=False, shortcut=True,
                 input_rescaling=True, loss_rescaling=True, lambda_reg=0., training_batch_size=100,
                 optimizer='Adam', learning_rate=5e-4, optimizer_kwargs=None,
                 learning_rate_decay_period=5, learning_rate_decay_gamma=0.9,
                 validation_fraction=0.15, stop_after_epochs=20):
        self.quantiles = np.asarray(quantiles)
        self.lows = np.atleast_1d(lows)
        self.highs = np.atleast_1d(highs)
        assert self.quantiles.ndim == 1 and np.all(np.diff(quantiles) > 0.)
        assert np.all(self.quantiles > 0.) and np.all(self.quantiles < 1.)
        assert self.lows.ndim == 1 and self.lows.shape == self.highs.shape
        self.quantile_method = str(quantile_method)
        if self.quantile_method == 'cumsum':
            self.binary_depth = 0
        elif self.quantile_method == 'binary':
            self.binary_depth = np.log2(self.quantiles.size + 1)
            assert np.isclose(self.binary_depth, int(self.binary_depth), rtol=0., atol=1e-7)
            self.binary_depth = int(self.binary_depth)
        else:
            raise ValueError
        self.quantiles_01 = np.concatenate([[0.], self.quantiles, [1.]])
        self.theta_size = self.lows.size
        self.alpha = float(alpha)
        self.loss = QuantileLoss(self.quantiles, self.alpha)
        self.add_fc = add_fc
        self.nonlinear = nonlinear
        self.batch_norm = bool(batch_norm)
        self.shortcut = bool(shortcut)
        self.input_rescaling = bool(input_rescaling)
        self.loss_rescaling = bool(loss_rescaling)
        self.lambda_reg = float(lambda_reg)
        self.training_batch_size = training_batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if optimizer_kwargs is not None:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            self.optimizer_kwargs = {}
        self.learning_rate_decay_period = learning_rate_decay_period
        self.learning_rate_decay_gamma = learning_rate_decay_gamma
        self.validation_fraction = validation_fraction
        self.stop_after_epochs = stop_after_epochs
        self.mlp_list = []

    def set_dim(self, x):
        self.mlp_list = []
        self.x_size = 0 if x is None else x.shape[-1]
        if self.quantile_method == 'cumsum':
            out_features = self.quantiles.size + 1
        elif self.quantile_method == 'binary':
            out_features = 2**(self.binary_depth + 1) - 2
        else:
            raise RuntimeError
        for i in range(self.theta_size):
            if i == 0 and self.x_size == 0:
                self.mlp_list.append(None)
            else:
                self.mlp_list.append(QuantileMLP1D(
                    self.lows[i], self.highs[i], self.quantile_method, self.binary_depth,
                    self.x_size + i, out_features, self.add_fc, self.nonlinear, self.batch_norm,
                    self.shortcut
                ))

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def train(self, theta, x=None):
        theta = torch.as_tensor(theta, dtype=torch.float)
        if theta.ndim == 1:
            theta = theta[:, None]
        assert theta.ndim == 2 and theta.shape[-1] == self.theta_size
        if x is not None:
            x = torch.as_tensor(x, dtype=torch.float)
            assert x.ndim == 2
            assert theta.shape[0] == x.shape[0]
        self.set_dim(x)
        self.train_loss = []
        self.valid_loss = []
        for i in range(self.theta_size):
            if i == 0 and x is None:
                knots = np.quantile(theta[:, 0].detach().numpy(), self.quantiles)
                self.mlp_list[0] = self.interp_1d(knots, 0)
                self.mlp_list[0].set_all()
                self.train_loss.append(None)
                self.valid_loss.append(None)
            else:
                n_epoch, train_loss, valid_loss = self.train_one(i, theta, x, self.mlp_list[i])
                print(f'Finished training QuantileMLP1D for dim #{i} after {n_epoch} epoches, valid'
                      f' loss = {valid_loss[-1][0]:.5f} + {valid_loss[-1][1]:.5f} = '
                      f'{valid_loss[-1][2]:.5f}.')
                self.train_loss.append(train_loss)
                self.valid_loss.append(valid_loss)
        return self.train_loss, self.valid_loss

    def train_one(self, i, theta, x, model):
        n_all = theta.shape[0]
        n_train = int((1 - self.validation_fraction) * n_all)
        n_valid = n_all - n_train
        if i > 0 and x is not None:
            in_tensor = torch.concat((x, theta[:, :i]), axis=1)
        elif i > 0 and x is None:
            in_tensor = theta[:, :i]
        elif i == 0 and x is not None:
            in_tensor = x
        else:
            raise RuntimeError
        out_tensor = theta[:, i]
        if self.input_rescaling:
            model.set_whitening(in_tensor)
        if self.loss_rescaling:
            out_std = torch.std(out_tensor).detach()
            out_std = out_std if out_std > 0. else 1.
        else:
            out_std = 1.

        class TrainData(Dataset):
            def __len__(self):
                return n_train
            def __getitem__(self, i):
                return in_tensor[i], out_tensor[i]

        class ValidData(Dataset):
            def __len__(self):
                return n_valid
            def __getitem__(self, i):
                return in_tensor[n_train + i], out_tensor[n_train + i]

        train_data = TrainData()
        valid_data = ValidData()
        train_loader = DataLoader(dataset=train_data, batch_size=self.training_batch_size,
                                  num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=self.training_batch_size,
                                  num_workers=1, pin_memory=True, shuffle=False, drop_last=False)
        optimizer = eval('torch.optim.' + self.optimizer)
        optimizer = optimizer(model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.learning_rate_decay_period,
            gamma=self.learning_rate_decay_gamma, verbose=False
        )
        loss_train_all = []
        loss_valid_all = []
        i_epoch = 0

        while not self.check_convergence(loss_valid_all):
            model.train()
            loss_train_0 = 0.
            loss_train_1 = 0.
            loss_train = 0.
            for j, (in_now, out_now) in enumerate(train_loader):
                # x_now = x_now.to(device)
                # theta_now = theta_now.to(device)
                # print(in_now.dtype)
                y = model(in_now, return_raw=True)
                loss_0 = self.loss(y[0], out_now) / out_std
                loss_1 = self.lambda_reg * torch.mean(y[1]**2) if self.lambda_reg > 0. else torch.tensor(0.)
                loss_now = loss_0 + loss_1
                optimizer.zero_grad()
                loss_now.backward()
                optimizer.step()
                loss_train_0 += loss_0.detach().numpy() * in_now.shape[0]
                loss_train_1 += loss_1.detach().numpy() * in_now.shape[0]
                loss_train += loss_now.detach().numpy() * in_now.shape[0]
            loss_train_0 /= len(train_data)
            loss_train_1 /= len(train_data)
            loss_train /= len(train_data)
            assert np.isfinite(loss_train)
            loss_train_all.append([loss_train_0, loss_train_1, loss_train])

            model.eval()
            loss_valid_0 = 0.
            loss_valid_1 = 0.
            loss_valid = 0.
            with torch.no_grad():
                for j, (in_now, out_now) in enumerate(valid_loader):
                    # x_now = x_now.to(device)
                    # theta_now = theta_now.to(device)
                    y = model(in_now, return_raw=True)
                    loss_0 = self.loss(y[0], out_now) / out_std
                    loss_1 = self.lambda_reg * torch.mean(y[1]**2) if self.lambda_reg > 0. else torch.tensor(0.)
                    loss_now = loss_0 + loss_1
                    loss_valid_0 += loss_0.detach().numpy() * in_now.shape[0]
                    loss_valid_1 += loss_1.detach().numpy() * in_now.shape[0]
                    loss_valid += loss_now.detach().numpy() * in_now.shape[0]
                loss_valid_0 /= len(valid_data)
                loss_valid_1 /= len(valid_data)
                loss_valid /= len(valid_data)
                assert np.isfinite(loss_valid)
                loss_valid_all.append([loss_valid_0, loss_valid_1, loss_valid])

            scheduler.step()
            i_epoch += 1
        return i_epoch, np.asarray(loss_train_all), np.asarray(loss_valid_all)

    def interp_1d(self, knots, i):
        knots_01 = np.concatenate([self.lows[i:(i + 1)], knots, self.highs[i:(i + 1)]])
        return Interp1D(knots_01, self.quantiles_01).set_all()

    def sample(self, n, x=None, random_seed=None):
        with torch.no_grad():
            if x is None:
                assert self.x_size == 0
                theta_all = torch.as_tensor(
                    self.mlp_list[0].sample(n, random_seed), dtype=torch.float)[:, None]
            else:
                x = torch.atleast_2d(torch.as_tensor(x, dtype=torch.float))
                assert tuple(x.shape) == (1, self.x_size)
                self.mlp_list[0].eval()
                knots_now = self.mlp_list[0](x).detach().numpy().flatten()
                theta_all = torch.as_tensor(
                    self.interp_1d(knots_now, 0).sample(n, random_seed), dtype=torch.float)[:, None]
            for i in range(1, self.theta_size):
                if x is None:
                    input_now = theta_all
                else:
                    input_now = torch.concat(
                        (torch.tile(x, (theta_all.shape[0], 1)), theta_all), axis=1)
                self.mlp_list[i].eval()
                knots_now = self.mlp_list[i](input_now).detach().numpy()
                theta_now = torch.as_tensor(
                    np.concatenate([self.interp_1d(k, i).sample(1) for k in knots_now]),
                    dtype=torch.float)
                theta_all = torch.concat((theta_all, theta_now[:, None]), axis=1)
            return theta_all.detach()

    def check_convergence(self, valid_loss):
        valid_loss = np.asarray(valid_loss)
        if (len(valid_loss) > self.stop_after_epochs and
            np.nanmin(valid_loss[:-self.stop_after_epochs, 2]) <=
            np.nanmin(valid_loss[-self.stop_after_epochs:, 2])):
            return True
        else:
            return False
