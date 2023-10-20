import numpy as np
import torch
from torch import nn
from typing import Any, Mapping
from .interp import Interp1D, broaden
from scipy.stats import norm, chi2
import warnings

__all__ = ['MLP', 'QuantileNet1D', 'QuantileInterp1D', 'QuantileNet', 'get_quantile_net']


def _set_cdfs_pred(cdfs_pred):
    if isinstance(cdfs_pred, int):
        assert cdfs_pred >= 2
        cdfs_pred = np.linspace(0, 1, cdfs_pred + 1)[1:-1]
    else:
        try:
            cdfs_pred = np.asarray(cdfs_pred, dtype=float).reshape(-1)
            assert np.all(cdfs_pred >= 0.)
            assert np.all(cdfs_pred <= 1.)
            assert np.all(np.diff(cdfs_pred) > 0.)
        except Exception:
            raise ValueError
    return cdfs_pred


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
        self.register_buffer('mu_x', torch.tensor(0.))
        self.register_buffer('sigma_x', torch.tensor(1.))
        self.register_buffer('mu_theta', torch.tensor(0.))
        self.register_buffer('sigma_theta', torch.tensor(1.))

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
        x = x.contiguous()

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

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # TODO: temp fix, should figure out what's happening
        # without this, it will only load the first dim of these buffer vars
        super(MLP, self).load_state_dict(state_dict, strict)
        self.mu_x = state_dict['mu_x']
        self.sigma_x = state_dict['sigma_x']
        self.mu_theta = state_dict['mu_theta']
        self.sigma_theta = state_dict['sigma_theta']


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
    cdfs_pred : int or array_like of float, optional
        The CDFs corresponding to the quantiles you want to predict. If ``int``, will divide the
        interval ``[0, 1]`` into ``cdfs_pred`` bins and therefore fit the evenly spaced
        ``cdfs_pred - 1`` quantiles between ``0`` (exclusive) and ``1`` (exclusive). Otherwise,
        should be in ascending order, larger than 0, and smaller than 1. Set to ``16`` by default.
    quantile_method : str, optional
        Should be either ``'cumsum'`` or ``'binary'``. Note that ``'binary'`` is not well tested at
        the moment.
    binary_depth : int, optional
        The depth of binary tree. Only used if ``'quantile_method'`` is ``'binary'``.
    p_tail_limit : float, optional
        Lower bound of the tail pdf in the one-side boundary interpolation scheme. Set to ``0.7``
        by default.
    split_threshold : float, optional
        The threshold for splitting into two peaks to account for multimodality during the
        interpolation. Set to ``1e-2`` by default.
    kwargs : dict, optional
        Additional keyword arguments to be passed to ``MLP``. Note that the ``output_neurons``
        parameter will be automatically set according to ``cdfs_pred``.

    Notes
    -----
    See ``MLP`` for the additional parameters, some of which are required by the initializer.
    """
    def __init__(self, i, low, high, cdfs_pred=16, quantile_method='cumsum', binary_depth=0,
                 p_tail_limit=0.6, split_threshold=1e-2, **kwargs):
        self.cdfs_pred = _set_cdfs_pred(cdfs_pred)
        kwargs['output_neurons'] = self.cdfs_pred.size + 1
        super(QuantileNet1D, self).__init__(**kwargs)
        self.i = int(i)
        self.low = float(low)
        self.high = float(high)
        self.cdfs = np.concatenate([[0.], self.cdfs_pred, [1.]])
        self.quantile_method = str(quantile_method)
        if self.quantile_method not in ('cumsum', 'binary'):
            raise ValueError
        self.binary_depth = int(binary_depth)
        self.p_tail_limit = float(p_tail_limit)
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
        knots_pred : 1-d or 2-d array_like of float
            The locations of predicted quantiles to be interpolated.
        """
        knots_pred = np.atleast_2d(knots_pred)
        assert knots_pred.ndim == 2
        knots = np.concatenate([np.full((knots_pred.shape[0], 1), self.low),
                                knots_pred,
                                np.full((knots_pred.shape[0], 1), self.high)], axis=1)
        return Interp1D(knots=knots, cdfs=self.cdfs, p_tail_limit=self.p_tail_limit,
                        split_threshold=self.split_threshold)

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, d=None, batch_size=None,
               device='cpu', broadening_factor=None):
        random_seed = np.random.default_rng(random_seed)
        i = self.i
        d = 1 if d is None else int(d)
        with torch.no_grad():
            self.to(device)
            self.eval()
            n, x, theta = _check_n_x_theta(n, x, theta)
            if theta is None and x is None:
                raise ValueError('x and theta cannot both be None.')
            if batch_size is not None and n > batch_size >= 1:
                batch_size = int(batch_size)
                return np.concatenate(
                    [self.sample(n=min(batch_size, n - i * batch_size),
                                 x=(x if (x is None or x.shape[0] <= 1) else
                                    x[(i * batch_size):((i + 1) * batch_size)]),
                                 theta=(theta if (theta is None or theta.shape[0] <= 1) else
                                        theta[(i * batch_size):((i + 1) * batch_size)]),
                                 random_seed=random_seed, sobol=sobol, d=d, batch_size=batch_size,
                                 device=device, broadening_factor=broadening_factor)
                     for i in range(int(np.ceil(n / batch_size)))]
                )
            else:
                n, x, theta = _broadcast_batch(n, x, theta)
                if x is not None:
                    x = x.contiguous().to(device)
                if theta is not None:
                    theta = theta.contiguous().to(device)
                knots_pred = self(x, theta).detach().cpu().numpy()
                return self.interp_1d(knots_pred).sample(n=n, random_seed=random_seed, sobol=sobol,
                                                         i=i, d=d,
                                                         broadening_factor=broadening_factor)

    def _f_interp(self, x, theta, batch_size, device, target, **kwargs):
        i = self.i
        with torch.no_grad():
            self.to(device)
            self.eval()
            _, x, theta = _check_n_x_theta(None, x, theta)
            if theta is None:
                raise ValueError('theta cannot be None.')
            if theta.shape[1] < i + 1:
                raise ValueError('invalid shape for theta.')
            max_x_theta = max(x.shape[0] if x is not None else -np.inf,
                              theta.shape[0] if theta is not None else -np.inf)
            if batch_size is not None and max_x_theta > batch_size >= 1:
                batch_size = int(batch_size)
                return np.concatenate(
                    [self._f_interp(x=(x if (x is None or x.shape[0] <= 1) else
                                       x[(i * batch_size):((i + 1) * batch_size)]),
                                    theta=(theta if (theta is None or theta.shape[0] <= 1) else
                                           theta[(i * batch_size):((i + 1) * batch_size)]),
                                    batch_size=batch_size, device=device, target=target,
                                    **kwargs)
                     for i in range(int(np.ceil(theta.shape[0] / batch_size)))]
                )
            else:
                _, x, theta = _broadcast_batch(None, x, theta)
                if x is not None:
                    x = x.contiguous().to(device)
                if i > 0:
                    theta_prev = theta[:, :i].contiguous().to(device)
                elif i == 0:
                    theta_prev = None
                else:
                    raise RuntimeError('invalid value for i.')
                theta_now = theta[:, i].detach().cpu().numpy().astype(np.float64)
                knots_pred = self(x, theta_prev).detach().cpu().numpy().astype(np.float64)
                if target == 'pdf':
                    return self.interp_1d(knots_pred).pdf(x=theta_now, **kwargs)
                elif target == 'cdf':
                    return self.interp_1d(knots_pred).cdf(x=theta_now, **kwargs)
                else:
                    raise ValueError('invalid value for target.')

    def pdf(self, x=None, theta=None, batch_size=None, device='cpu', broadening_factor=None):
        return self._f_interp(x=x, theta=theta, batch_size=batch_size, device=device, target='pdf',
                              broadening_factor=broadening_factor)

    def cdf(self, x=None, theta=None, local=False, batch_size=None, device='cpu',
            broadening_factor=None):
        return self._f_interp(x=x, theta=theta, batch_size=batch_size, device=device, target='cdf',
                              local=local, broadening_factor=broadening_factor)

    def qm_latent(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                  broadening_factor=None):
        if qm_method.lower() == 'cdf':
            return self.cdf(x=x, theta=theta, local=False, batch_size=batch_size, device=device,
                            broadening_factor=broadening_factor)
        elif qm_method.lower() == 'cdf_local':
            return self.cdf(x=x, theta=theta, local=True, batch_size=batch_size, device=device,
                            broadening_factor=broadening_factor)
        else:
            raise NotImplementedError('currently only cdf and cdf_local are implemented for '
                                      'ref_dist.')

    def qm_rank(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                broadening_factor=None, qm_latent=None, ref_dist='gaussian'):
        if qm_latent is None:
            qm_latent = self.qm_latent(x=x, theta=theta, qm_method=qm_method, batch_size=batch_size,
                                       device=device, broadening_factor=broadening_factor)
        if isinstance(ref_dist, str) and ref_dist.lower() == 'gaussian':
            return chi2.cdf(norm.ppf(qm_latent)**2, df=1)
        else:
            raise NotImplementedError('currently only gaussian is implemented for ref_dist.')


def _check_n_x_theta(n, x, theta):
    if theta is not None:
        theta = torch.as_tensor(theta, dtype=torch.float)
        if theta.ndim == 0:
            theta = theta[None, None]
        elif theta.ndim == 1:
            warnings.warn('theta.ndim == 1 is ambiguous here and may lead to unexpected behavior, '
                          'please consider giving me a 2-dim Tensor.', RuntimeWarning)
            theta = theta[None]
        elif theta.ndim == 2:
            pass
        else:
            raise ValueError('theta should be a 2-dim Tensor.')
    if x is not None:
        x = torch.as_tensor(x, dtype=torch.float)

    if n is None:
        if x is not None and theta is not None:
            if ((x.shape[0] not in (1, theta.shape[0])) and
                (theta.shape[0] not in (1, x.shape[0]))):
                warnings.warn('the shapes of x and theta seem inconsistent, assuming they all '
                              'share the same x.', RuntimeWarning)
                x = x[None]
    else:
        n = int(n)
        if theta is not None:
            if theta.shape[0] not in (1, n):
                raise ValueError('the shape of theta is inconsistent with n.')
        if x is not None:
            if x.shape[0] not in (1, n):
                warnings.warn('the shape of x is inconsistent with n, assuming they all share the '
                              'same x.', RuntimeWarning)
                x = x[None]

    return n, x, theta


def _broadcast_batch(n, x, theta):
    if x is not None and theta is not None:
        if n is None:
            if x.shape[0] == theta.shape[0]:
                pass
            elif x.shape[0] == 1 and theta.shape[0] > 1:
                x = torch.tile(x, [theta.shape[0]] + list(np.ones(x.ndim - 1, dtype=int)))
            elif theta.shape[0] == 1 and x.shape[0] > 1:
                theta = torch.tile(theta, [x.shape[0]] + list(np.ones(theta.ndim - 1, dtype=int)))
            else:
                raise ValueError('the shapes of x and theta do not match.')
        else:
            if x.shape[0] == n:
                pass
            elif x.shape[0] == 1:
                if theta.shape[0] == n:
                    x = torch.tile(x, [n] + list(np.ones(x.ndim - 1, dtype=int)))
            else:
                raise ValueError('invalid shape for x.')
            if theta.shape[0] == n:
                pass
            elif theta.shape[0] == 1:
                if x.shape[0] == n:
                    theta = torch.tile(theta, [n] + list(np.ones(theta.ndim - 1, dtype=int)))
            else:
                raise ValueError('invalid shape for theta.')
    return n, x, theta


class QuantileInterp1D(Interp1D):
    """
    Convenience class for the first dimension of theta when x is None.

    No NNs are used since the quantiles can be directly estimated from the emperical values.

    Parameters
    ----------
    theta : 1-d array_like of float
        The first dimension of theta.
    low : float
        The lower bound of prior.
    high : float
        The upper bound of prior.
    cdfs_pred : int or array_like of float, optional
        The CDFs corresponding to the quantiles you want to predict. If ``int``, will divide the
        interval ``[0, 1]`` into ``cdfs_pred`` bins and therefore fit the evenly spaced
        ``cdfs_pred - 1`` quantiles between ``0`` (exclusive) and ``1`` (exclusive). Otherwise,
        should be in ascending order, larger than 0, and smaller than 1. Set to ``16`` by default.
    p_tail_limit : float, optional
        Lower bound of the tail pdf in the one-side boundary interpolation scheme. Set to ``0.7``
        by default.
    split_threshold : float, optional
        The threshold for splitting into two peaks to account for multimodality during the
        interpolation. Set to ``1e-2`` by default.
    """
    def __init__(self, theta, low, high, cdfs_pred=16, p_tail_limit=0.6, split_threshold=1e-2,
                 configs=None):
        self.i = 0
        if configs is None:
            if isinstance(theta, torch.Tensor):
                theta = theta.detach().cpu().numpy()
            cdfs_pred = _set_cdfs_pred(cdfs_pred)
            knots_pred = np.quantile(theta, cdfs_pred)
            super(QuantileInterp1D, self).__init__(
                knots=np.concatenate([[low], knots_pred, [high]]),
                cdfs=np.concatenate([[0.], cdfs_pred, [1.]]),
                p_tail_limit=p_tail_limit,
                split_threshold=split_threshold
            )
        else:
            super(QuantileInterp1D, self).__init__(configs=configs)

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, i=None, d=None,
               batch_size=None, device='cpu', broadening_factor=None):
        return Interp1D.sample(self, n=n, random_seed=random_seed, sobol=sobol, i=i, d=d,
                               broadening_factor=broadening_factor)

    def _check_theta(self, theta):
        try:
            if isinstance(theta, torch.Tensor):
                theta = np.atleast_1d(np.ascontiguousarray(theta.detach().cpu().numpy(),
                                                           dtype=np.float64))
            else:
                theta = np.atleast_1d(np.ascontiguousarray(theta, dtype=np.float64))
            if theta.ndim == 1:
                pass
            elif theta.ndim == 2:
                theta = theta[:, self.i].copy()
            else:
                raise ValueError('invalid dim for theta.')
        except Exception:
            raise ValueError('invalid value for theta.')
        return theta

    def pdf(self, x=None, theta=None, batch_size=None, device='cpu', broadening_factor=None):
        theta = self._check_theta(theta)
        return Interp1D.pdf(self, x=theta, broadening_factor=broadening_factor)

    def cdf(self, x=None, theta=None, local=False, batch_size=None, device='cpu',
            broadening_factor=None):
        theta = self._check_theta(theta)
        return Interp1D.cdf(self, x=theta, local=local, broadening_factor=broadening_factor)

    def qm_latent(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                  broadening_factor=None):
        if qm_method.lower() == 'cdf':
            return self.cdf(x=x, theta=theta, local=False, batch_size=batch_size, device=device,
                            broadening_factor=broadening_factor)
        elif qm_method.lower() == 'cdf_local':
            return self.cdf(x=x, theta=theta, local=True, batch_size=batch_size, device=device,
                            broadening_factor=broadening_factor)
        else:
            raise NotImplementedError('currently only cdf and cdf_local are implemented for '
                                      'ref_dist.')

    def qm_rank(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                broadening_factor=None, qm_latent=None, ref_dist='gaussian'):
        if qm_latent is None:
            qm_latent = self.qm_latent(x=x, theta=theta, qm_method=qm_method, batch_size=batch_size,
                                       device=device, broadening_factor=broadening_factor)
        if isinstance(ref_dist, str) and ref_dist.lower() == 'gaussian':
            return chi2.cdf(norm.ppf(qm_latent)**2, df=1)
        else:
            raise NotImplementedError('currently only gaussian is implemented for ref_dist.')

    def broaden(self, broadening_factor=1.1):
        return QuantileInterp1D(configs=broaden(self.configs, broadening_factor, self.p_tail_limit,
                                                self.split_threshold))


class _QuantileInterp1D(QuantileInterp1D, nn.Module):
    """
    Placeholder for potentially-not-yet-fitted QuantileInterp1D.
    """
    def __init__(self, low, high, cdfs_pred=16, p_tail_limit=0.6, split_threshold=1e-2):
        self.i = 0
        self.low = low
        self.high = high
        self.cdfs_pred = cdfs_pred
        self.p_tail_limit = p_tail_limit
        self.split_threshold = split_threshold
        self._ok = False
        nn.Module.__init__(self)
        self._placeholder_module = nn.ReLU() # not used, only to make this a valid nn.Module

    def fit(self, theta):
        QuantileInterp1D.__init__(self, theta=theta, low=self.low, high=self.high,
                                  cdfs_pred=self.cdfs_pred, p_tail_limit=self.p_tail_limit,
                                  split_threshold=self.split_threshold)


# TODO: fall back to prior when extreme values make softmax fail
class QuantileNet(nn.ModuleList):
    """
    List of individual 1-dim conditional quantile networks.
    """
    def __init__(self, modules):
        if (hasattr(modules, '__iter__') and len(modules) >= 1 and
            all(isinstance(_, (QuantileNet1D, QuantileInterp1D, _QuantileInterp1D)) or _ is None
                for _ in modules)):
            super(QuantileNet, self).__init__(modules)
        else:
            raise ValueError

    def check(self):
        for i in range(len(self)):
            if not (self[i].i == i and
                    isinstance(self[i], (QuantileNet1D, QuantileInterp1D, _QuantileInterp1D))):
                return False
        return True

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, batch_size=None,
               device='cpu', broadening_factor=None):
        # theta is not used
        n, x, _ = _check_n_x_theta(n, x, None)
        random_seed = np.random.default_rng(random_seed)
        if not self.check():
            raise RuntimeError('This QuantileNet is not well defined.')
        with torch.no_grad():
            if batch_size is not None and n > batch_size >= 1:
                batch_size = int(batch_size)
                return np.concatenate(
                    [self.sample(n=min(batch_size, n - i * batch_size),
                                 x=(x if (x is None or x.shape[0] <= 1) else
                                    x[(i * batch_size):((i + 1) * batch_size)]),
                                 theta=None, random_seed=random_seed, sobol=sobol,
                                 batch_size=batch_size, device=device,
                                 broadening_factor=broadening_factor)
                     for i in range(int(np.ceil(n / batch_size)))]
                )
            else:
                theta_all = self[0].sample(n=n, x=x, random_seed=random_seed, sobol=sobol,
                                           d=len(self), batch_size=batch_size, device=device,
                                           broadening_factor=broadening_factor)[:, None]
                for i in range(1, len(self)):
                    theta_now = self[i].sample(n=n, x=x, theta=theta_all, random_seed=random_seed,
                                               sobol=sobol, d=len(self), batch_size=batch_size,
                                               device=device,
                                               broadening_factor=broadening_factor)[: None]
                    theta_all = np.concatenate((theta_all, theta_now[:, None]), axis=1)
                return theta_all

    def pdf(self, x=None, theta=None, batch_size=None, device='cpu', broadening_factor=None):
        return np.prod([s.pdf(x=x, theta=theta, batch_size=batch_size, device=device,
                              broadening_factor=broadening_factor) for s in self], axis=0)

    def qm_latent(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                  broadening_factor=None):
        return np.concatenate([s.qm_latent(x=x, theta=theta, qm_method=qm_method,
                                           batch_size=batch_size, device=device,
                                           broadening_factor=broadening_factor)[:, None]
                               for s in self], axis=1)

    def qm_rank(self, x=None, theta=None, qm_method='cdf_local', batch_size=None, device='cpu',
                broadening_factor=None, qm_latent=None, ref_dist='gaussian'):
        if qm_latent is None:
            qm_latent = self.qm_latent(x=x, theta=theta, qm_method=qm_method, batch_size=batch_size,
                                       device=device, broadening_factor=broadening_factor)
        if isinstance(ref_dist, str) and ref_dist.lower() == 'gaussian':
            return chi2.cdf(np.sum(norm.ppf(qm_latent)**2, axis=1), df=len(self))
        else:
            raise NotImplementedError('currently only gaussian is implemented for ref_dist.')


def get_quantile_net(low, high, input_neurons, hidden_neurons, i_start=None, i_end=None,
                     cdfs_pred=16, p_tail_limit=0.6, split_threshold=1e-2, activation='relu',
                     batch_norm=False, shortcut=True, embedding_net=None):
    low = np.asarray(low)
    high = np.asarray(high)
    if not (low.shape == high.shape and low.ndim == 1):
        raise ValueError
    module_list = []
    for i in range(low.size):
        if (i_start is None or i >= i_start) and (i_end is None or i < i_end):
            if input_neurons == 0 and i == 0:
                module_list.append(_QuantileInterp1D(low=low[0], high=high[0], cdfs_pred=cdfs_pred,
                                                     p_tail_limit=p_tail_limit,
                                                     split_threshold=split_threshold))
            else:
                if isinstance(embedding_net, (list, tuple)):
                    embedding_net_now = embedding_net[i]
                else:
                    embedding_net_now = embedding_net
                module_list.append(QuantileNet1D(
                    i=i, low=low[i], high=high[i], cdfs_pred=cdfs_pred, p_tail_limit=p_tail_limit,
                    split_threshold=split_threshold, input_neurons=input_neurons + i,
                    hidden_neurons=hidden_neurons, activation=activation, batch_norm=batch_norm,
                    shortcut=shortcut, embedding_net=embedding_net_now
                ))
        else:
            module_list.append(None)
    return QuantileNet(module_list)
