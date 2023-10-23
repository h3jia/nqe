import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from collections import namedtuple
import warnings
from .qnet import _set_cdfs_pred, _QuantileInterp1D, QuantileNet1D

__all__ = ['QuantileLoss', 'train_1d', 'TrainResult']


class QuantileLoss:
    """
    Weighted L1 loss for quantile prediction.

    Parameters
    ----------
    cdfs_pred : int or array_like of float, optional
        The CDFs corresponding to the quantiles you want to predict. If ``int``, will divide the
        interval ``[0, 1]`` into ``cdfs_pred`` bins and therefore fit the evenly spaced
        ``cdfs_pred - 1`` quantiles between ``0`` (exclusive) and ``1`` (exclusive). Otherwise,
        should be in ascending order, larger than 0, and smaller than 1. Set to ``16`` by default.
    device : str, optional
        The device on which you train the model. Set to ``'cpu'`` by default.
    """
    def __init__(self, cdfs_pred, device='cpu'):
        self.cdfs_pred = torch.as_tensor(_set_cdfs_pred(cdfs_pred), dtype=torch.float).to(device)
        self.device = device

    def __call__(self, input, target, p0=1., p0_weights=None, p0_replacement=True):
        # in_now shape: # of points, (# of data dims + # of previous theta dims)
        # input = model(in_now) shape: # of points, # of cdfs
        # target = out_now shape: # of points
        if target.ndim == input.ndim:
            pass
        elif target.ndim == input.ndim - 1:
            target = target[..., None]
        else:
            raise RuntimeError
        weights = torch.where(target > input, self.cdfs_pred, 1. - self.cdfs_pred)
        results_raw = torch.abs(weights * (input - target)) # (# of points, # of cdfs)
        if not (p0 == 1. and p0_weights is None):
            n0 = int(p0 * results_raw.shape[-1])
            if p0_weights is None:
                p0_weights = torch.ones_like(self.cdfs_pred)
            p0_weights = torch.as_tensor(p0_weights).detach().to(self.device)
            try:
                i0 = torch.multinomial(p0_weights, n0, replacement=p0_replacement)
            except Exception:
                warnings.warn('p0 multinomial sampling failed, trying equal weights for now')
                i0 = torch.multinomial(torch.ones_like(self.cdfs_pred).detach().to(self.device),
                                       n0, replacement=p0_replacement)
            if i0.ndim == 1:
                results_raw = results_raw[..., i0]
            elif i0.ndim == 2:
                results_raw = torch.gather(results_raw, 1, i0)
            else:
                raise RuntimeError(f'i0.ndim should be 1 or 2 instead of {i0.ndim}')
        return torch.mean(results_raw)


# TODO: freeze the embedding network
def train_1d(quantile_net_1d, device='cpu', x=None, theta=None, batch_size=100,
             validation_fraction=0.15, train_loader=None, valid_loader=None, rescale_data=False,
             p0=1., f0=0., p0_weights=None, p0_replacement=True, p0_batch_avg=True,
             p0_after_epochs=0, lambda_reg=0., f1=1.1, f2=0.8, custom_l1=None, l1_after_epochs=0,
             optimizer='Adam', learning_rate=5e-4, optimizer_kwargs=None, scheduler='StepLR',
             learning_rate_decay_period=5, learning_rate_decay_gamma=0.9, scheduler_kwargs=None,
             stop_after_epochs=20, stop_tol=1e-4, max_epochs=200, return_best_epoch=True,
             verbose=True):
    if isinstance(quantile_net_1d, _QuantileInterp1D): # for the first dim without x, no nn required
        if theta is not None:
            theta = np.asarray(theta, dtype=np.float64)
            assert theta.ndim == 2
            theta_0 = theta[:, 0]
        elif train_loader is not None or valid_loader is not None:
            theta_all = []
            if train_loader is not None:
                for batch_now in train_loader:
                    x_now, theta_now = _decode_batch(batch_now, 'cpu')
                    assert theta_now.ndim == 2
                    theta_all.append(theta_now[:, 0])
            if valid_loader is not None:
                for batch_now in valid_loader:
                    x_now, theta_now = _decode_batch(batch_now, 'cpu')
                    assert theta_now.ndim == 2
                    theta_all.append(theta_now[:, 0])
            theta_0 = torch.concat(theta_all).detach().cpu().numpy().astype(np.float64)
        else:
            raise ValueError("you didn't give me the data for training.")
        quantile_net_1d.fit(theta_0)
        if verbose:
            print(f'finished fitting the emperical quantiles for dim 0')
        return TrainResult(state_dict=quantile_net_1d.configs, l0_train=None, l1_train=None,
                           l0_valid=None, l1_valid=None, lambda_reg=None, i_epoch=None)

    elif isinstance(quantile_net_1d, QuantileNet1D):
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
                raise ValueError("you didn't give me the data for training.")

        if rescale_data:
            mu_x = []
            sigma_x = []
            mu_theta = []
            sigma_theta = []

            for batch_now in train_loader:
                x_now, theta_now = _decode_batch(batch_now, device)
                if x_now is not None:
                    mu_x.append(torch.mean(x_now, dim=0, keepdim=True))
                    sigma_x.append(torch.std(x_now, dim=0, keepdim=True))
                if quantile_net_1d.i > 0:
                    mu_theta.append(torch.mean(theta_now[..., :quantile_net_1d.i], dim=0,
                                               keepdim=True))
                    sigma_theta.append(torch.std(theta_now[..., :quantile_net_1d.i], dim=0,
                                                 keepdim=True))

            mu_x = torch.mean(torch.concat(mu_x), dim=0) if len(mu_x) > 0 else None
            sigma_x = torch.mean(torch.concat(sigma_x)**2, dim=0)**0.5 if len(sigma_x) > 0 else None
            mu_theta = torch.mean(torch.concat(mu_theta), dim=0) if len(mu_theta) > 0 else None
            sigma_theta = (torch.mean(torch.concat(sigma_theta)**2, dim=0)**0.5 if
                           len(sigma_theta) > 0 else None)
            # print(mu_x, sigma_x, mu_theta, sigma_theta)
            quantile_net_1d.set_rescaling(mu_x=mu_x, sigma_x=sigma_x, mu_theta=mu_theta,
                                          sigma_theta=sigma_theta)

        loss = QuantileLoss(quantile_net_1d.cdfs_pred, device=device)
        cdfs_01 = np.concatenate([[0.], quantile_net_1d.cdfs_pred, [1.]])
        dcdf = torch.as_tensor(cdfs_01[1:] - cdfs_01[:-1], dtype=torch.float).to(device)
        log_dcdf = torch.log(dcdf)

        if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            optimizer = optimizer(quantile_net_1d.parameters(), **optimizer_kwargs)
        elif isinstance(optimizer, torch.optim.Optimizer):
            pass
        elif isinstance(optimizer, str):
            optimizer = eval('torch.optim.' + optimizer)
            optimizer = optimizer(quantile_net_1d.parameters(), lr=learning_rate,
                                  **optimizer_kwargs)
        else:
            raise ValueError

        if isinstance(scheduler, type) and issubclass(scheduler,
                                                      torch.optim.lr_scheduler.LRScheduler):
            scheduler = scheduler(optimizer, **scheduler_kwargs)
        elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            pass
        elif scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=learning_rate_decay_period, gamma=learning_rate_decay_gamma,
                verbose=False
            )
        elif isinstance(scheduler, str):
            scheduler = eval('torch.optim.lr_scheduler.' + optimizer)
            scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            raise ValueError

        l0_train_all = []
        l1_train_all = []
        l0_valid_all = []
        l1_valid_all = []
        i_epoch_all = []
        i_epoch = -1
        lambda_reg_now = 0.
        state_dict_cache = []

        while not _check_convergence(l0_valid_all, l1_valid_all, lambda_reg_now, stop_after_epochs,
                                     stop_tol, max_epochs):
            i_epoch += 1
            i_epoch_all.append(i_epoch)
            lambda_reg_now = lambda_reg if i_epoch >= l1_after_epochs else 0.
            quantile_net_1d.train()
            l0_train = 0.
            l1_train = 0.
            n_theta_now = 0
            for j, batch_now in enumerate(train_loader):
                x_now, theta_now = _decode_batch(batch_now, device)
                if quantile_net_1d.i > 0:
                    y_now = quantile_net_1d(x_now, theta_now[..., :quantile_net_1d.i],
                                            return_raw=True)
                else:
                    y_now = quantile_net_1d(x_now, None, return_raw=True)
                if i_epoch >= p0_after_epochs:
                    p0_now = p0
                    if not (p0 == 1. and f0 == 0. and p0_weights is None):
                        p0_weights_now = dcdf / torch.softmax(y_now[1], axis=-1)
                        p0_weights_now = 0.5 * (p0_weights_now[..., 1:] + p0_weights_now[..., :-1])
                        if p0_batch_avg:
                            p0_weights_now = torch.mean(p0_weights_now, axis=0)**(-f0)
                        else:
                            p0_weights_now = p0_weights_now**(-f0)
                else:
                    p0_now = 1.
                    p0_weights_now = None
                l0_now = loss(y_now[0], theta_now[..., quantile_net_1d.i], p0=p0_now,
                              p0_weights=p0_weights_now, p0_replacement=p0_replacement)
                if custom_l1 is not None:
                    l1_now = custom_l1(y_now[1])
                else:
                    assert log_dcdf.shape[0] >= 3
                    logp_bin = log_dcdf - y_now[1]
                    logp_bin_c = logp_bin[..., 1:-1]
                    logp_bin_l = logp_bin[..., :-2]
                    logp_bin_r = logp_bin[..., 2:]
                    logp_bin_lr = torch.concat((logp_bin_l[None], logp_bin_r[None]), axis=0)
                    logp_bin_max = torch.max(logp_bin_lr, axis=0)[0]
                    # logp_bin_min = torch.min(logp_bin_lr, axis=0)[0]
                    # if 0. < f1 < 1.:
                    #     logp_bin_mean = torch.logsumexp(
                    #         torch.concat((np.log(f1) + logp_bin_max[None],
                    #                       np.log(1. - f1) + logp_bin_min[None]),
                    #                      axis=0), axis=0)
                    # elif f1 >= 1.:
                    #     logp_bin_mean = np.log(f1) + logp_bin_max
                    # elif f1 <= 0.:
                    #     logp_bin_mean = np.log(1. - f1) + logp_bin_min
                    # else:
                    #     raise ValueError(f'invalid value f1 = {f1}')
                    # logp_bin_mean = torch.log(f1 * torch.exp(logp_bin_max) +
                    #                           (1 - f1) * torch.exp(logp_bin_min))
                    logp_bin_mean = torch.clip(np.log(0.5 * f1) + torch.logsumexp(logp_bin_lr,
                                                                                  axis=0),
                                               np.log(f2) + logp_bin_max, None)
                    _tmp = logp_bin_c - logp_bin_mean
                    l1_2 = torch.where(_tmp > 0., _tmp**2, 0.)
                    l1_now = torch.mean(torch.sum(l1_2, axis=-1))
                loss_now = l0_now * (1 + lambda_reg_now * l1_now) if lambda_reg_now else l0_now
                optimizer.zero_grad()
                loss_now.backward()
                optimizer.step()
                l0_train += l0_now.detach().cpu().numpy() * theta_now.shape[0]
                l1_train += l1_now.detach().cpu().numpy() * theta_now.shape[0]
                n_theta_now += theta_now.shape[0]
            l0_train /= n_theta_now
            l1_train /= n_theta_now
            if not np.isfinite(l0_train):
                raise RuntimeError(f'l0_train = {l0_train} is not finite')
            if not np.isfinite(l1_train):
                raise RuntimeError(f'l1_train = {l1_train} is not finite')
            l0_train_all.append(l0_train)
            l1_train_all.append(l1_train)

            quantile_net_1d.eval()
            l0_valid = 0.
            l1_valid = 0.
            n_theta_now = 0
            with torch.no_grad():
                for j, batch_now in enumerate(valid_loader):
                    x_now, theta_now = _decode_batch(batch_now, device)
                    if quantile_net_1d.i > 0:
                        y_now = quantile_net_1d(x_now, theta_now[..., :quantile_net_1d.i],
                                                return_raw=True)
                    else:
                        y_now = quantile_net_1d(x_now, None, return_raw=True)
                    l0_now = loss(y_now[0], theta_now[..., quantile_net_1d.i])
                    if custom_l1 is not None:
                        l1_now = custom_l1(y_now[1])
                    else:
                        assert log_dcdf.shape[0] >= 3
                        logp_bin = log_dcdf - y_now[1]
                        logp_bin_c = logp_bin[..., 1:-1]
                        logp_bin_l = logp_bin[..., :-2]
                        logp_bin_r = logp_bin[..., 2:]
                        logp_bin_lr = torch.concat((logp_bin_l[None], logp_bin_r[None]), axis=0)
                        logp_bin_max = torch.max(logp_bin_lr, axis=0)[0]
                        # logp_bin_min = torch.min(logp_bin_lr, axis=0)[0]
                        # if 0. < f1 < 1.:
                        #     logp_bin_mean = torch.logsumexp(
                        #         torch.concat((np.log(f1) + logp_bin_max[None],
                        #                       np.log(1. - f1) + logp_bin_min[None]),
                        #                      axis=0), axis=0)
                        # elif f1 >= 1.:
                        #     logp_bin_mean = np.log(f1) + logp_bin_max
                        # elif f1 <= 0.:
                        #     logp_bin_mean = np.log(1. - f1) + logp_bin_min
                        # else:
                        #     raise ValueError(f'invalid value f1 = {f1}')
                        # logp_bin_mean = torch.log(f1 * torch.exp(logp_bin_max) +
                        #                           (1 - f1) * torch.exp(logp_bin_min))
                        logp_bin_mean = torch.clip(np.log(0.5 * f1) + torch.logsumexp(logp_bin_lr,
                                                                                      axis=0),
                                                   np.log(f2) + logp_bin_max, None)
                        _tmp = logp_bin_c - logp_bin_mean
                        l1_2 = torch.where(_tmp > 0., _tmp**2, 0.)
                        l1_now = torch.mean(torch.sum(l1_2, axis=-1))
                    # loss_now = l0_now * (1 + lambda_reg_now * l1_now)
                    l0_valid += l0_now.detach().cpu().numpy() * theta_now.shape[0]
                    l1_valid += l1_now.detach().cpu().numpy() * theta_now.shape[0]
                    n_theta_now += theta_now.shape[0]
                l0_valid /= n_theta_now
                l1_valid /= n_theta_now
                if not np.isfinite(l0_train):
                    raise RuntimeError(f'l0_train = {l0_train} is not finite')
                if not np.isfinite(l1_train):
                    raise RuntimeError(f'l1_train = {l1_train} is not finite')
                l0_valid_all.append(l0_valid)
                l1_valid_all.append(l1_valid)

            if return_best_epoch:
                state_dict_cache.append(deepcopy(quantile_net_1d.state_dict()))
                if len(state_dict_cache) > stop_after_epochs + 1:
                    state_dict_cache = state_dict_cache[-(stop_after_epochs + 1):]
            scheduler.step()
            if verbose > 0 and (i_epoch + 1) % verbose == 0:
                print(f'finished epoch {i_epoch + 1}, l0_train = {l0_train:.5f}, '
                      f'l1_train = {l1_train:.5f}, l0_valid = {l0_valid:.5f}, '
                      f'l1_valid = {l1_valid:.5f}')

        if return_best_epoch:
            i_epoch_cache = i_epoch_all[-len(state_dict_cache):]
            l0_valid_cache = l0_valid_all[-len(state_dict_cache):]
            l1_valid_cache = l1_valid_all[-len(state_dict_cache):]
            loss_valid_cache = np.asarray(l0_valid_cache) * (
                1 + lambda_reg_now * np.asarray(l1_valid_cache))
            i_best_cache = np.argmin(loss_valid_cache)
            state_dict = state_dict_cache[i_best_cache]
            quantile_net_1d.load_state_dict(state_dict)
            i_epoch = i_epoch_cache[i_best_cache]
        else:
            state_dict = deepcopy(quantile_net_1d.state_dict())
        # state_dict = {k: v.cpu() for k, v in state_dict.items()}

        if verbose > 0:
            print(f'finished training dim {quantile_net_1d.i}, '
                  f'l0_valid_best = {np.asarray(l0_valid_all)[i_epoch]:.5f}, '
                  f'l1_valid_best = {np.asarray(l1_valid_all)[i_epoch]:.5f}')
        return TrainResult(state_dict=state_dict, l0_train=np.asarray(l0_train_all),
                           l1_train=np.asarray(l1_train_all), l0_valid=np.asarray(l0_valid_all),
                           l1_valid=np.asarray(l1_valid_all), lambda_reg=lambda_reg_now,
                           i_epoch=i_epoch)

    else:
        raise ValueError("I don't know how to train this quantile_net_1d.")


TrainResult = namedtuple('TrainResult', ['state_dict', 'l0_train', 'l1_train', 'l0_valid',
                                         'l1_valid', 'lambda_reg', 'i_epoch'])


def _decode_batch(batch_now, device):
    if isinstance(batch_now, torch.Tensor):
        return None, batch_now.to(device, torch.float)
    elif hasattr(batch_now, '__iter__') and len(batch_now) == 2:
        return batch_now[0].to(device, torch.float), batch_now[1].to(device, torch.float)
    else:
        raise ValueError


def _check_convergence(l0_valid_all, l1_valid_all, lambda_reg, stop_after_epochs, stop_tol,
                       max_epochs):
    if len(l0_valid_all) >= max_epochs:
        return True
    elif len(l0_valid_all) <= stop_after_epochs:
        return False
    else:
        loss_all = np.asarray(l0_valid_all) * (1 + lambda_reg * np.asarray(l1_valid_all))
        # if np.nanmin(loss_all[:-stop_after_epochs]) <= np.nanmin(loss_all[-stop_after_epochs:]):
        if loss_all[-(stop_after_epochs + 1)] <= (1 + stop_tol) * np.nanmin(
            loss_all[-stop_after_epochs:]):
            return True
        else:
            return False
