import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from .interp import Interp1D
from copy import deepcopy
from collections import namedtuple
import warnings

__all__ = ['QuantileLoss', 'MLP', 'QuantileNet1D', 'QuantileInterp1D', 'QuantileNet',
           'get_quantile_net', 'train_1d', 'TrainResult']


def _set_quantiles_pred(quantiles_pred):
    if isinstance(quantiles_pred, int):
        quantiles_pred = np.linspace(0, 1, quantiles_pred + 1)[1:-1]
    else:
        try:
            quantiles_pred = np.asarray(quantiles_pred, dtype=float).reshape(-1)
            assert np.all(quantiles_pred > 0.)
            assert np.all(quantiles_pred < 1.)
            assert np.all(np.diff(quantiles_pred) > 0.)
        except Exception:
            raise ValueError
    return quantiles_pred


class QuantileLoss:
    """
    Weighted L1 loss for quantile prediction.

    Parameters
    ----------
    quantiles_pred : 1-d array_like
        The quantiles you want to predict, should be larger than 0 and smaller than 1.
    alpha : float, optional
        Each term in the loss will be weighted by ``exp(alpha * abs(quantile - 0.5))``. Set to
        ``0.`` by default.
    device : str, optional
        The device on which you train the model. Set to ``'cpu'`` by default.
    """
    def __init__(self, quantiles_pred, alpha=0., device='cpu'):
        self.quantiles_pred = torch.as_tensor(quantiles_pred, dtype=torch.float).to(device)
        self.alpha = float(alpha)
        self._weights = (
            torch.exp(self.alpha * torch.abs(self.quantiles_pred - 0.5))[None]).to(device)

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
        weights = torch.where(target > input, self.quantiles_pred, 1. - self.quantiles_pred)
        return torch.mean(torch.abs(weights * self._weights * (input - target)))


class MLP(nn.Module):
    """
    Basic MLP module.

    Parameters
    ----------
    input_neurons : int
        The number of input neurons, which should match the dimension of data `plus` any additional
        predictor variables. When ``embedding_net`` is not ``None``, this should match the `output`
        dimension of ``embedding_net`` plus the dimension of any additional predictor variables.
    output_neurons : int
        The number of output neurons.
    hidden_neurons : None, int, 1-d array_like of int
        The number(s) of neurons in the hidden layer(s). If ``None``, no hidden layers will be
        added.
    activation : str or nn.Module, optional
        The non-linear activation function. If ``str``, should be among ``'tanh', 'relu',
        'leakyrelu', 'elu'``. Set to ``'relu'`` by default.
    batch_norm : bool, optional
        Whether to add batch normalization before each non-linear activation function. Set to
        ``False`` by default.
    shortcut : bool, optional
        Whether to add shortcut connections, i.e. the input layer will be concatenated with each
        hidden layer. Set to ``True`` by default.
    embedding_net : None or nn.Module, optional
        Additional embedding network before the MLP layers. Set to ``None`` by default, which is
        interpreted as ``F(x, theta) = x``. The input should be the data x and the variables theta;
        the output should be a flattened 1-dim tensor.
    """
    def __init__(self, input_neurons, output_neurons, hidden_neurons, activation='relu',
                 batch_norm=False, shortcut=True, embedding_net=None):
        super(MLP, self).__init__()
        self._make_fc(input_neurons, output_neurons, hidden_neurons, activation, batch_norm,
                      shortcut)
        if isinstance(embedding_net, nn.Module) or embedding_net is None:
            self.embedding_net = embedding_net
        else:
            raise ValueError
        self.train_loss = []
        self.valid_loss = []
        self.mu_x = 0.
        self.sigma_x = 1.
        self.mu_theta = 0.
        self.sigma_theta = 1.

    def _make_fc(self, input_neurons, output_neurons, hidden_neurons, activation, batch_norm,
                 shortcut):
        if isinstance(activation, nn.Module):
            self.activation = activation
        elif isinstance(activation, str):
            if activation.lower() == 'tanh':
                self.activation = nn.Tanh()
            elif activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'leakyrelu':
                self.activation = nn.LeakyReLU()
            elif activation.lower() == 'elu':
                self.activation = nn.ELU()
            else:
                raise ValueError
        else:
            raise ValueError
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.batch_norm = bool(batch_norm)
        self.shortcut = bool(shortcut)
        if hidden_neurons is None:
            self.fc_layers.append(nn.Linear(input_neurons, output_neurons))
        else:
            hidden_neurons = list(hidden_neurons)
            hidden_neurons.insert(0, input_neurons)
            hidden_neurons.append(output_neurons)
            self.fc_layers.append(nn.Linear(hidden_neurons[0], hidden_neurons[1]))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_neurons[1]))
            # pass thru activation, concat with shortcut
            k = hidden_neurons[0] if shortcut else 0
            for i in range(1, len(hidden_neurons) - 2):
                self.fc_layers.append(nn.Linear(hidden_neurons[i] + k, hidden_neurons[i + 1]))
                if batch_norm:
                    self.bn_layers.append(nn.BatchNorm1d(hidden_neurons[i + 1]))
                # pass thru activation, concat with shortcut
            self.fc_layers.append(nn.Linear(hidden_neurons[-2] + k, hidden_neurons[-1]))

    def set_rescaling(self, x=None, mu_x=None, sigma_x=None, theta=None, mu_theta=None,
                      sigma_theta=None):
        """
        Set the optional rescaling.
        """
        if x is not None:
            self.mu_x = torch.mean(x, dim=0).detach()
            self.sigma_x = torch.std(x, dim=0)
            self.sigma_x = torch.where(self.sigma_x > 0., self.sigma_x, 1e-8).detach()
        if mu_x is not None:
            self.mu_x = torch.as_tensor(mu_x)
        if sigma_x is not None:
            self.sigma_x = torch.as_tensor(sigma_x)
        if theta is not None:
            self.mu_theta = torch.mean(theta, dim=0).detach()
            self.sigma_theta = torch.std(theta, dim=0)
            self.sigma_theta = torch.where(self.sigma_theta > 0., self.sigma_theta, 1e-8).detach()
        if mu_theta is not None:
            self.mu_theta = torch.as_tensor(mu_theta)
        if sigma_theta is not None:
            self.sigma_theta = torch.as_tensor(sigma_theta)

    def _forward(self, x=None, theta=None):
        """
        The forward evaluation of the model.

        Parameters
        ----------
        x : Tensor or None, optional
            The input variables to be first passed to the embedding network. Set to ``None`` by
            default.
        theta : Tensor or None, optional
            The input variables to be first passed to the embedding network as well as directly
            passed to the MLP. Set to ``None`` by default.

        Notes
        -----
        ``x`` and ``theta`` cannot both be None.
        """
        if x is not None and theta is not None:
            x = (x - self.mu_x) / self.sigma_x
            theta = (theta - self.mu_theta) / self.sigma_theta
            x = self.embedding_net(x, theta) if self.embedding_net is not None else x
            x = torch.concat((x, theta), dim=-1)
        elif x is not None and theta is None:
            x = (x - self.mu_x) / self.sigma_x
            x = self.embedding_net(x, theta) if self.embedding_net is not None else x
        elif x is None and theta is not None:
            theta = (theta - self.mu_theta) / self.sigma_theta
            x = theta
        else:
            raise ValueError

        if len(self.fc_layers) == 1:
            return self.fc_layers[0](x)
        elif len(self.fc_layers) > 1:
            if self.shortcut:
                x_clone = torch.clone(x)
            x = self.fc_layers[0](x)
            if self.batch_norm:
                x = self.bn_layers[0](x)
            x = self.activation(x)
            for i in range(1, len(self.fc_layers) - 1):
                if self.shortcut:
                    x = torch.concat((x, x_clone), axis=-1)
                x = self.fc_layers[i](x)
                if self.batch_norm:
                    x = self.bn_layers[i](x)
                x = self.activation(x)
            if self.shortcut:
                x = torch.concat((x, x_clone), axis=-1)
            return self.fc_layers[-1](x)
        else:
            raise RuntimeError

    forward = _forward

    __call__ = _forward


class QuantileNet1D(MLP):
    """
    Neural Network to predict the 1-dim quantiles.

    Parameters
    ----------
    i : int
        The index of theta predicted by this network.
    low : float
        The lower bound of prior.
    high : float
        The upper bound of prior.
    quantiles_pred : int or array_like of float, optional
        The quantiles to fit. If ``int``, will divide the interval ``[0, 1]`` into
        ``quantiles_pred`` bins and therefore fit the evenly spaced ``quantiles_pred - 1`` quantiles
        between ``0`` (exclusive) and ``1`` (exclusive). Otherwise, should be in ascending order,
        larger than 0, and smaller than 1. Set to ``12`` by default.
    quantile_method : str, optional
        Should be either ``'cumsum'`` or ``'binary'``. Note that ``'binary'`` is not well tested at
        the moment.
    binary_depth : int, optional
        The depth of binary tree. Only used if ``'quantile_method'`` is ``'binary'``.
    split_threshold : float, optional
        The threshold for splitting into two peaks to account for multimodality during the
        interpolation. Set to ``1e-2`` by default.
    kwargs : dict, optional
        Additional keyword arguments to be passed to ``MLP``. Note that the ``output_neurons``
        parameter will be automatically set according to ``quantiles_pred``.

    Notes
    -----
    See ``MLP`` for the additional parameters, some of which are required by the initializer.
    """
    def __init__(self, i, low, high, quantiles_pred=12, quantile_method='cumsum', binary_depth=0,
                 split_threshold=1e-2, **kwargs):
        self.quantiles_pred = _set_quantiles_pred(quantiles_pred)
        kwargs['output_neurons'] = self.quantiles_pred.size + 1
        super(QuantileNet1D, self).__init__(**kwargs)
        self.i = int(i)
        self.low = float(low)
        self.high = float(high)
        self.quantiles = np.concatenate([[0.], self.quantiles_pred, [1.]])
        self.quantile_method = str(quantile_method)
        if self.quantile_method not in ('cumsum', 'binary'):
            raise ValueError
        self.binary_depth = int(binary_depth)
        self.split_threshold = float(split_threshold)

    def forward(self, x=None, theta=None, return_raw=False):
        """
        The forward evaluation of the model.

        Parameters
        ----------
        x : Tensor or None, optional
            The input variables to be first passed to the embedding network. Set to ``None`` by
            default.
        theta : Tensor or None, optional
            The input variables to be first passed to the embedding network as well as directly
            passed to the MLP. Set to ``None`` by default.
        return_raw : bool, optional
            If False, only return the quantiles. If True, also return the `raw` output (the
            intermediate output before the softmax layer) which is typically used for
            regularization. Set to ``False`` by default.

        Notes
        -----
        ``x`` and ``theta`` cannot both be None, and should be consistent with the embedding
        network architecture.
        """
        x = self._forward(x, theta)
        if self.quantile_method == 'cumsum':
            y = self.low + (self.high - self.low) * torch.cumsum(torch.softmax(x, axis=-1),
                                                                 axis=-1)[..., :-1]
            return (y, x) if return_raw else y
        elif self.quantile_method == 'binary':
            # EXPERIMENTAL
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
                                                             2**(self.binary_depth - i - 1),
                                                             -1)
                    offset += 1
            y = self.low + (self.high - self.low) * torch.cumsum(y, axis=-1)[..., :-1]
            return (y, x) if return_raw else y
        else:
            return ValueError

    __call__ = forward

    def interp_1d(self, knots_pred):
        """
        Utility for the 1-dim quantile interpolation.

        Parameters
        ----------
        knots_pred : 1-d array_like of float
            The locations of predicted quantiles to be interpolated.
        """
        knots_pred = np.asarray(knots_pred)
        assert knots_pred.ndim == 1
        knots = np.concatenate([[self.low], knots_pred, [self.high]])
        return Interp1D(knots, self.quantiles, self.split_threshold).set_all()

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, batch_size=None,
               device='cpu'):
        random_seed = np.random.default_rng(random_seed)
        with torch.no_grad():
            self.to(device)
            self.eval()
            assert x is not None
            x = torch.as_tensor(x, dtype=torch.float)[None].to(device)
            if theta is None:
                if batch_size is not None and n > batch_size:
                    batch_size = int(batch_size)
                    assert batch_size > 0
                    return np.concatenate(
                        [self.sample(n=min(batch_size, n - i * batch_size), x=x, theta=None,
                                     random_seed=random_seed, sobol=sobol, batch_size=batch_size,
                                     device=device) for i in range(int(np.ceil(n / batch_size)))]
                    )
                else:
                    knots_pred = self(x, theta).detach().cpu().numpy()[0]
                    return self.interp_1d(knots_pred).sample(n=n, random_seed=random_seed,
                                                             sobol=sobol)
            else:
                theta = torch.atleast_2d(torch.as_tensor(theta, dtype=torch.float)).to(device)
                assert theta.ndim == 2
                assert theta.shape[0] == n
                if batch_size is not None and n > batch_size:
                    batch_size = int(batch_size)
                    assert batch_size > 0
                    return np.concatenate(
                        [self.sample(n=min(batch_size, n - i * batch_size), x=x,
                                     theta=theta[(i * batch_size):((i + 1) * batch_size)],
                                     random_seed=random_seed, sobol=sobol, batch_size=batch_size,
                                     device=device) for i in range(int(np.ceil(n / batch_size)))]
                    )
                else:
                    x = torch.tile(x, [n] + list(np.ones(x.ndim - 1, dtype=int)))
                    knots_pred = self(x, theta).detach().cpu().numpy()
                    return np.concatenate([
                        self.interp_1d(k).sample(n=1, random_seed=random_seed, sobol=sobol) for k in
                        knots_pred])


class QuantileInterp1D(Interp1D):
    """
    Convenience class for the first dimension of theta when x is None.

    No NNs are used since the quantiles can be directly estimated from the emperical values.

    Parameters
    ----------
    theta : 1_d array_like of float
        The first dimension of theta.
    low : float
        The lower bound of prior.
    high : float
        The upper bound of prior.
    quantiles_pred : int or array_like of float, optional
        The quantiles to fit. If ``int``, will divide the interval ``[0, 1]`` into
        ``quantiles_pred`` bins and therefore fit the evenly spaced ``quantiles_pred - 1`` quantiles
        between ``0`` (exclusive) and ``1`` (exclusive). Otherwise, should be in ascending order,
        larger than 0, and smaller than 1. Set to ``12`` by default.
    split_threshold : float, optional
        The threshold for splitting into two peaks to account for multimodality during the
        interpolation. Set to ``1e-2`` by default.
    """
    def __init__(self, theta, low, high, quantiles_pred=12, split_threshold=1e-2):
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        self.i = 0
        quantiles_pred = _set_quantiles_pred(quantiles_pred)
        knots_pred = np.quantile(theta, quantiles_pred)
        super(QuantileInterp1D, self).__init__(np.concatenate([[low], knots_pred, [high]]),
                                               np.concatenate([[0.], self.quantiles_pred, [1.]]),
                                               split_threshold)


class QuantileNet(nn.ModuleList):
    """
    List of individual 1-dim conditional quantile networks.
    """
    def __init__(self, modules):
        if (hasattr(modules, '__iter__') and len(modules) >= 1 and
            all(isinstance(_, (QuantileNet1D, QuantileInterp1D)) or _ is None for _ in modules)):
            super(QuantileNet, self).__init__(modules)
        else:
            raise ValueError

    def check(self):
        for i in range(len(self)):
            if not (self[i].i == i and isinstance(self[i], (QuantileNet1D, QuantileInterp1D))):
                return False
        return True

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, batch_size=None,
               device='cpu'):
        random_seed = np.random.default_rng(random_seed)
        if not self.check():
            raise RuntimeError('This QuantileNet is not well defined.')
        with torch.no_grad():
            if batch_size is not None and n > batch_size:
                batch_size = int(batch_size)
                assert batch_size > 0
                return np.concatenate(
                    [self.sample(n=min(batch_size, n - i * batch_size), x=x, theta=None,
                                 random_seed=random_seed, sobol=sobol, batch_size=batch_size,
                                 device=device) for i in range(int(np.ceil(n / batch_size)))]
                )
            else:
                theta_all = self[0].sample(n=n, x=x, random_seed=random_seed, sobol=sobol,
                                           batch_size=batch_size, device=device)[:, None]
                for i in range(1, len(self)):
                    theta_now = self[i].sample(n=n, x=x, theta=theta_all, random_seed=random_seed,
                                               sobol=sobol, batch_size=batch_size,
                                               device=device)[: None]
                    theta_all = np.concatenate((theta_all, theta_now[:, None]), axis=1)
                return theta_all


def get_quantile_net(low, high, input_neurons, hidden_neurons, i_start=None, i_end=None,
                     quantiles_pred=12, split_threshold=1e-2, activation='relu', batch_norm=False,
                     shortcut=True, embedding_net=None):
    low = np.asarray(low)
    high = np.asarray(high)
    if not (low.shape == high.shape and low.ndim == 1):
        raise ValueError
    module_list = []
    for i in range(low.size):
        if (i_start is None or i >= i_start) and (i_end is None or i < i_end):
            if isinstance(embedding_net, (list, tuple)):
                embedding_net_now = embedding_net[i]
            else:
                embedding_net_now = embedding_net
            module_list.append(QuantileNet1D(
                i=i, low=low[i], high=high[i], quantiles_pred=quantiles_pred,
                split_threshold=split_threshold, input_neurons=input_neurons + i,
                hidden_neurons=hidden_neurons, activation=activation, batch_norm=batch_norm,
                shortcut=shortcut, embedding_net=embedding_net_now
            ))
        else:
            module_list.append(None)
    return QuantileNet(module_list)


def train_1d(quantile_net_1d, device='cpu', x=None, theta=None, batch_size=100,
             validation_fraction=0.15, train_loader=None, valid_loader=None, alpha=0.,
             rescale_data=False, target_loss_ratio=0., beta_reg=0.5, drop_edge=False,
             lambda_max_factor=3., initial_max_ratio=0.1, initial_ratio_epochs=10, optimizer='Adam',
             learning_rate=5e-4, optimizer_kwargs=None, scheduler='StepLR',
             learning_rate_decay_period=5, learning_rate_decay_gamma=0.9, scheduler_kwargs=None,
             stop_after_epochs=20, stop_tol=1e-3, max_epochs=200, return_best_epoch=True,
             verbose=True):
    quantile_net_1d.to(device)
    if verbose is True:
        verbose = 5
    elif verbose is False:
        verbose = 0
    else:
        verbose = int(verbose)
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    if theta is not None:
        theta = torch.as_tensor(theta)
        if not theta.ndim == 2:
            raise ValueError
        n_all = theta.shape[0]
        n_train = int((1 - validation_fraction) * n_all)
        n_valid = n_all - n_train

        if x is None:
            class TrainData(Dataset):
                def __len__(self):
                    return n_train
                def __getitem__(self, i):
                    return theta[i]
            class ValidData(Dataset):
                def __len__(self):
                    return n_valid
                def __getitem__(self, i):
                    return theta[n_train + i]
        else:
            x = torch.as_tensor(x)
            if not theta.shape[0] == x.shape[0]:
                raise ValueError
            class TrainData(Dataset):
                def __len__(self):
                    return n_train
                def __getitem__(self, i):
                    return x[i], theta[i]
            class ValidData(Dataset):
                def __len__(self):
                    return n_valid
                def __getitem__(self, i):
                    return x[n_train + i], theta[n_train + i]

        train_data = TrainData()
        valid_data = ValidData()
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False)

    else:
        if not (isinstance(train_loader, DataLoader) and isinstance(valid_loader, DataLoader)):
            raise ValueError

    if rescale_data:
        mu_x = []
        sigma_x = []
        mu_theta = []
        sigma_theta = []

        for batch_now in train_loader:
            x_now, theta_now = _decode_batch(batch_now, device)
            if x_now is not None:
                mu_x.append(torch.mean(x_now, dim=0))
                sigma_x.append(torch.std(x_now, dim=0))
            if quantile_net_1d.i > 0:
                mu_theta.append(torch.mean(theta_now[..., :quantile_net_1d.i], dim=0))
                sigma_theta.append(torch.std(theta_now[..., :quantile_net_1d.i], dim=0))

        mu_x = torch.mean(torch.concat(mu_x), dim=0) if len(mu_x) > 0 else None
        sigma_x = torch.mean(torch.concat(sigma_x), dim=0) if len(sigma_x) > 0 else None
        mu_theta = torch.mean(torch.concat(mu_theta), dim=0) if len(mu_theta) > 0 else None
        sigma_theta = torch.mean(torch.concat(sigma_theta), dim=0) if len(sigma_theta) > 0 else None
        quantile_net_1d.set_rescaling(mu_x=mu_x, sigma_x=sigma_x, mu_theta=mu_theta,
                                      sigma_theta=sigma_theta)

    loss = QuantileLoss(quantile_net_1d.quantiles_pred, alpha, device=device)

    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(quantile_net_1d.parameters(), **optimizer_kwargs)
    elif isinstance(optimizer, torch.optim.Optimizer):
        pass
    elif isinstance(optimizer, str):
        optimizer = eval('torch.optim.' + optimizer)
        optimizer = optimizer(quantile_net_1d.parameters(), lr=learning_rate, **optimizer_kwargs)
    else:
        raise ValueError

    if isinstance(scheduler, type) and issubclass(scheduler, torch.optim.lr_scheduler.LRScheduler):
        scheduler = scheduler(optimizer, **scheduler_kwargs)
    elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        pass
    elif scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_period,
                                                    gamma=learning_rate_decay_gamma, verbose=False)
    elif isinstance(scheduler, str):
        scheduler = eval('torch.optim.lr_scheduler.' + optimizer)
        scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        raise ValueError

    l0_train_all = []
    l1_train_all = []
    l0_valid_all = []
    l1_valid_all = []
    lambda_reg_all = []
    i_epoch = -1
    lambda_reg = 0.
    state_dict_cache = []

    while not _check_convergence(l0_valid_all, l1_valid_all, lambda_reg_all, stop_after_epochs,
                                 stop_tol, max_epochs):
        i_epoch += 1
        lambda_reg_all.append(lambda_reg)
        quantile_net_1d.train()
        l0_train = 0.
        l1_train = 0.
        if i_epoch + 1 <= initial_ratio_epochs:
            target_loss_ratio_now = min(target_loss_ratio, initial_max_ratio)
        else:
            target_loss_ratio_now = target_loss_ratio
        for j, batch_now in enumerate(train_loader):
            x_now, theta_now = _decode_batch(batch_now, device)
            if quantile_net_1d.i > 0:
                y_now = quantile_net_1d(x_now, theta_now[..., :quantile_net_1d.i], return_raw=True)
            else:
                y_now = quantile_net_1d(x_now, None, return_raw=True)
            l0_now = loss(y_now[0], theta_now[..., quantile_net_1d.i])
            if target_loss_ratio_now > 0.:
                if drop_edge:
                    l1_now = torch.mean(y_now[1][..., 1:-1]**2)
                else:
                    l1_now = torch.mean(y_now[1]**2)
            else:
                l1_now = torch.tensor(0.)
            loss_now = l0_now + lambda_reg * l1_now
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()
            l0_train += l0_now.detach().cpu().numpy() # * theta_now.shape[0]
            l1_train += l1_now.detach().cpu().numpy() # * theta_now.shape[0]
        l0_train /= (j + 1)
        l1_train /= (j + 1)
        assert np.isfinite(l0_train) and np.isfinite(l1_train)
        l0_train_all.append(l0_train)
        l1_train_all.append(l1_train)

        quantile_net_1d.eval()
        l0_valid = 0.
        l1_valid = 0.
        with torch.no_grad():
            for j, batch_now in enumerate(valid_loader):
                x_now, theta_now = _decode_batch(batch_now, device)
                if quantile_net_1d.i > 0:
                    y_now = quantile_net_1d(x_now, theta_now[..., :quantile_net_1d.i],
                                            return_raw=True)
                else:
                    y_now = quantile_net_1d(x_now, None, return_raw=True)
                l0_now = loss(y_now[0], theta_now[..., quantile_net_1d.i])
                if target_loss_ratio_now > 0.:
                    if drop_edge:
                        l1_now = torch.mean(y_now[1][..., 1:-1]**2)
                    else:
                        l1_now = torch.mean(y_now[1]**2)
                else:
                    l1_now = torch.tensor(0.)
                loss_now = l0_now + lambda_reg * l1_now
                l0_valid += l0_now.detach().cpu().numpy() * theta_now.shape[0]
                l1_valid += l1_now.detach().cpu().numpy() * theta_now.shape[0]
            l0_valid /= len(valid_loader.dataset)
            l1_valid /= len(valid_loader.dataset)
            assert np.isfinite(l0_valid) and np.isfinite(l1_valid)
            l0_valid_all.append(l0_valid)
            l1_valid_all.append(l1_valid)

        if target_loss_ratio_now > 0.:
            lambda_reg = ((1. - beta_reg) * lambda_reg + beta_reg * target_loss_ratio_now *
                          l0_valid / l1_valid)
            i_best_l0 = np.argmin(l0_valid_all)
            lambda_max = (lambda_max_factor * target_loss_ratio_now * l0_valid_all[i_best_l0] /
                          l1_valid_all[i_best_l0])
            if lambda_reg > lambda_max:
                warnings.warn(f'lambda_reg exceeds its max value {lambda_max:.5f}, please consider '
                              f'increasing lambda_max_factor or reducing target_loss_ratio',
                              RuntimeWarning)
                lambda_reg = lambda_max
        if return_best_epoch:
            state_dict_cache.append(deepcopy(quantile_net_1d.state_dict()))
            if len(state_dict_cache) > stop_after_epochs + 1:
                state_dict_cache = state_dict_cache[-(stop_after_epochs + 1):]
        scheduler.step()
        if verbose > 0 and (i_epoch + 1) % verbose == 0:
            print(f'finished epoch {i_epoch + 1}, l0_valid = {l0_valid:.5f}, l1_valid = '
                  f'{l1_valid:.5f}, lambda_reg = {lambda_reg:.5f}')

    if return_best_epoch and i_epoch + 1 < max_epochs:
        state_dict = state_dict_cache[0]
        quantile_net_1d.load_state_dict(state_dict)
        i_epoch -= stop_after_epochs
    else:
        state_dict = deepcopy(quantile_net_1d.state_dict())
    # state_dict={k: v.cpu() for k, v in state_dict.items()}

    if verbose > 0:
        print(f'finished training dim {quantile_net_1d.i}, l0_valid_best = '
              f'{np.asarray(l0_valid_all)[i_epoch]:.5f}, l1_valid_best = '
              f'{np.asarray(l1_valid_all)[i_epoch]:.5f}')
    return TrainResult(state_dict=state_dict, l0_train=np.asarray(l0_train_all),
                       l1_train=np.asarray(l1_train_all), l0_valid=np.asarray(l0_valid_all),
                       l1_valid=np.asarray(l1_valid_all), lambda_reg=np.asarray(lambda_reg_all),
                       i_epoch=i_epoch)


TrainResult = namedtuple('TrainResult', ['state_dict', 'l0_train', 'l1_train', 'l0_valid',
                                         'l1_valid', 'lambda_reg', 'i_epoch'])


def _decode_batch(batch_now, device):
    if isinstance(batch_now, torch.Tensor):
        return None, batch.to(device, torch.float)
    elif hasattr(batch_now, '__iter__') and len(batch_now) == 2:
        return batch_now[0].to(device, torch.float), batch_now[1].to(device, torch.float)
    else:
        raise ValueError


def _check_convergence(l0_valid_all, l1_valid_all, lambda_reg_all, stop_after_epochs, stop_tol,
                       max_epochs):
    if len(l0_valid_all) >= max_epochs:
        return True
    elif len(l0_valid_all) <= stop_after_epochs:
        return False
    else:
        loss_all = np.asarray(l0_valid_all) + lambda_reg_all[-1] * np.asarray(l1_valid_all)
        # if np.nanmin(loss_all[:-stop_after_epochs]) <= np.nanmin(loss_all[-stop_after_epochs:]):
        if loss_all[-(stop_after_epochs + 1)] <= (1 + stop_tol) * np.nanmin(
            loss_all[-stop_after_epochs:]):
            return True
        else:
            return False
