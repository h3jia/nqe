import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
from collections import namedtuple
import warnings
from .qnet import _set_cdfs_pred, _QuantileInterp1D, QuantileNet1D
import os
import datetime

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
                warnings.warn('p0 multinomial sampling failed, trying equal weights for now',
                              RuntimeWarning)
                i0 = torch.multinomial(torch.ones_like(self.cdfs_pred).detach().to(self.device),
                                       n0, replacement=p0_replacement)
            if i0.ndim == 1:
                results_raw = results_raw[..., i0]
            elif i0.ndim == 2:
                results_raw = torch.gather(results_raw, 1, i0)
            else:
                raise RuntimeError(f'i0.ndim should be 1 or 2 instead of {i0.ndim}')
        return torch.mean(results_raw)


# NOTE: p0_batch_avg is a bit tricky with multiple gpus, removed for now
# NOTE: p0_after_epochs and l1_after_epochs seem not quite useful, removed for now
# TODO: pretrain by first train median only, and then freeze the embedding network and
#       only train the quantiles, and finally train everything together?
def train_1d(quantile_net_1d, device='cpu', save_path=None, save_period=5,
             x=None, theta=None, batch_size=100, validation_fraction=0.15,
             train_loader=None, valid_loader=None, rescale_data=False,
             p0=0.5, f0=1., p0_weights=None, p0_replacement=False,
             lambda_reg=0.1, f1=1.1, f2=0.8, custom_l1=None,
             optimizer='Adam', learning_rate=5e-4, optimizer_kwargs=None,
             scheduler='DelayedStepLR', learning_rate_decay_delay=0, learning_rate_decay_period=5,
             learning_rate_decay_gamma=0.9, scheduler_kwargs=None,
             stop_after_epochs=20, stop_tol=1e-4, max_epochs=200, return_best_epoch=True,
             verbose=True):
    if isinstance(quantile_net_1d, _QuantileInterp1D): # for the first dim without x, no nn required
        if dist.is_initialized():
            warnings.warn('for now, with torch.distributed each rank will evaluate a separate '
                          'QuantileInterp1D and the model will not be synced across different '
                          'ranks.', RuntimeWarning)
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
            print(f'[{datetime.datetime.now()}]  finished fitting the emperical quantiles for '
                  f'dim 0', flush=True)
        return TrainResult(best_state=quantile_net_1d.configs, last_state=quantile_net_1d.configs,
                           lambda_reg=None, l0_train=None, l1_train=None, l0_valid=None,
                           l1_valid=None)

    elif isinstance(quantile_net_1d, QuantileNet1D) or isinstance(quantile_net_1d, DDP):
        if isinstance(quantile_net_1d, QuantileNet1D):
            quantile_net_1d.to(device)
            model = quantile_net_1d
            model_local = quantile_net_1d
            if dist.is_initialized():
                warnings.warn('quantile_net_1d should be a DDP model for proper distributed '
                              'training.', RuntimeWarning)
        elif isinstance(quantile_net_1d, DDP):
            model = quantile_net_1d
            model_local = quantile_net_1d.module
            if not dist.is_initialized():
                warnings.warn('quantile_net_1d is a DDP model but distributed training has not been'
                              ' initialized.', RuntimeWarning)
        else:
            raise RuntimeError

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
            if dist.is_initialized():
                warnings.warn('if you want to correctly use multiple GPUs with torch.distributed, '
                              'please directly give me the data loader.', RuntimeWarning)
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
            if dist.is_initialized():
                warnings.warn('for now the rescaling parameters will not be synced between multiple'
                              ' GPUs.', RuntimeWarning)
            mu_x = []
            sigma_x = []
            mu_theta = []
            sigma_theta = []

            for batch_now in train_loader:
                x_now, theta_now = _decode_batch(batch_now, device)
                if x_now is not None:
                    mu_x.append(torch.mean(x_now, dim=0, keepdim=True))
                    sigma_x.append(torch.std(x_now, dim=0, keepdim=True))
                if model_local.i > 0:
                    mu_theta.append(torch.mean(theta_now[..., :model_local.i], dim=0,
                                               keepdim=True))
                    sigma_theta.append(torch.std(theta_now[..., :model_local.i], dim=0,
                                                 keepdim=True))

            mu_x = torch.mean(torch.concat(mu_x), dim=0) if len(mu_x) > 0 else None
            sigma_x = torch.mean(torch.concat(sigma_x)**2, dim=0)**0.5 if len(sigma_x) > 0 else None
            mu_theta = torch.mean(torch.concat(mu_theta), dim=0) if len(mu_theta) > 0 else None
            sigma_theta = (torch.mean(torch.concat(sigma_theta)**2, dim=0)**0.5 if
                           len(sigma_theta) > 0 else None)
            # print(mu_x, sigma_x, mu_theta, sigma_theta)
            model_local.set_rescaling(mu_x=mu_x, sigma_x=sigma_x, mu_theta=mu_theta,
                                      sigma_theta=sigma_theta)

        loss = QuantileLoss(model_local.cdfs_pred, device=device)
        cdfs_01 = np.concatenate([[0.], model_local.cdfs_pred, [1.]])
        dcdf = torch.as_tensor(cdfs_01[1:] - cdfs_01[:-1], dtype=torch.float).to(device)
        log_dcdf = torch.log(dcdf)

        if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            optimizer = optimizer(model.parameters(), lr=learning_rate, **optimizer_kwargs)
        elif isinstance(optimizer, torch.optim.Optimizer):
            pass
        elif isinstance(optimizer, str):
            optimizer = eval('torch.optim.' + optimizer)
            optimizer = optimizer(model.parameters(), lr=learning_rate, **optimizer_kwargs)
        else:
            raise ValueError

        if isinstance(scheduler, type) and issubclass(scheduler,
                                                      torch.optim.lr_scheduler.LRScheduler):
            scheduler = scheduler(optimizer, **scheduler_kwargs)
        elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            pass
        elif scheduler == 'DelayedStepLR':
            lr_lambda = lambda x: (
                1. if (x < learning_rate_decay_delay or
                       (x - learning_rate_decay_delay) % learning_rate_decay_period)
                else learning_rate_decay_gamma
            )
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda,
                                                                  verbose=False)
        elif isinstance(scheduler, str):
            scheduler = eval('torch.optim.lr_scheduler.' + optimizer)
            scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            raise ValueError

        prev_state = _get_prev_state(save_path, device, verbose)
        if prev_state is None:
            l0_train = []
            l1_train = []
            l0_valid = []
            l1_valid = []
            best_state = None
        else:
            model_local.load_state_dict(prev_state.last_state.model)
            optimizer.load_state_dict(prev_state.last_state.optimizer)
            scheduler.load_state_dict(prev_state.last_state.scheduler)
            l0_train = list(prev_state.l0_train)
            l1_train = list(prev_state.l1_train)
            l0_valid = list(prev_state.l0_valid)
            l1_valid = list(prev_state.l1_valid)
            best_state = prev_state.best_state

        if dist.is_initialized():
            dist.barrier()

        while not _check_convergence(l0_valid, l1_valid, lambda_reg, stop_after_epochs, stop_tol,
                                     max_epochs):
            i_epoch = len(l0_valid)
            if dist.is_initialized():
                train_loader.sampler.set_epoch(i_epoch)
                valid_loader.sampler.set_epoch(i_epoch)
            # lambda_reg_now = lambda_reg if i_epoch >= l1_after_epochs else 0.

            model.train()
            l0_train_now = torch.tensor(0., device=device)
            l1_train_now = torch.tensor(0., device=device)
            n_now = torch.tensor(0, device=device)
            for j, batch_now in enumerate(train_loader):
                x_now, theta_now = _decode_batch(batch_now, device)
                if model_local.i > 0:
                    y_now = model(x_now, theta_now[..., :model_local.i], return_raw=True)
                else:
                    y_now = model(x_now, None, return_raw=True)

                # if i_epoch >= p0_after_epochs:
                p0_now = p0
                if not (p0 == 1. and f0 == 0. and p0_weights is None):
                    p0_weights_now = dcdf / torch.softmax(y_now[1], axis=-1)
                    p0_weights_now = 0.5 * (p0_weights_now[..., 1:] + p0_weights_now[..., :-1])
                    # if p0_batch_avg:
                    #     p0_weights_now = torch.mean(p0_weights_now, axis=0)**(-f0)
                    # else:
                    p0_weights_now = p0_weights_now**(-f0)
                else:
                    p0_weights_now = p0_weights
                # else:
                #     p0_now = 1.
                #     p0_weights_now = None

                l0_now = loss(y_now[0], theta_now[..., model_local.i], p0=p0_now,
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
                loss_now = l0_now * (1 + lambda_reg * l1_now) if lambda_reg else l0_now
                optimizer.zero_grad()
                loss_now.backward()
                optimizer.step()
                l0_train_now += l0_now.detach() * theta_now.shape[0]
                l1_train_now += l1_now.detach() * theta_now.shape[0]
                n_now += theta_now.shape[0]
            l0_train_now, l1_train_now = _compute_total_loss(l0_train_now, l1_train_now, n_now)
            if not np.isfinite(l0_train_now):
                raise RuntimeError(f'l0_train_now = {l0_train_now} is not finite')
            if not np.isfinite(l1_train_now):
                raise RuntimeError(f'l1_train_now = {l1_train_now} is not finite')
            l0_train.append(l0_train_now)
            l1_train.append(l1_train_now)

            model.eval()
            l0_valid_now = torch.tensor(0., device=device)
            l1_valid_now = torch.tensor(0., device=device)
            n_now = torch.tensor(0, device=device)
            with torch.no_grad():
                for j, batch_now in enumerate(valid_loader):
                    x_now, theta_now = _decode_batch(batch_now, device)
                    if model_local.i > 0:
                        y_now = model(x_now, theta_now[..., :model_local.i], return_raw=True)
                    else:
                        y_now = model(x_now, None, return_raw=True)
                    l0_now = loss(y_now[0], theta_now[..., model_local.i])
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
                    l0_valid_now += l0_now.detach() * theta_now.shape[0]
                    l1_valid_now += l1_now.detach() * theta_now.shape[0]
                    n_now += theta_now.shape[0]
                l0_valid_now, l1_valid_now = _compute_total_loss(l0_valid_now, l1_valid_now, n_now)
                if not np.isfinite(l0_valid_now):
                    raise RuntimeError(f'l0_valid_now = {l0_valid_now} is not finite')
                if not np.isfinite(l1_valid_now):
                    raise RuntimeError(f'l1_valid_now = {l1_valid_now} is not finite')
                l0_valid.append(l0_valid_now)
                l1_valid.append(l1_valid_now)

            valid_loss = _l0_lambda_l1(l0_valid, l1_valid, lambda_reg)
            if best_state is None or np.argmin(valid_loss) == valid_loss.size - 1:
                best_state = deepcopy(FullState(
                    model={k: v.cpu() for k, v in model_local.state_dict().items()},
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ))
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                if verbose > 0 and (i_epoch + 1) % verbose == 0:
                    print(f'[{datetime.datetime.now()}]  finished epoch {i_epoch + 1}, '
                          f'l0_train = {l0_train_now:.5f}, l1_train = {l1_train_now:.5f}, '
                          f'l0_valid = {l0_valid_now:.5f}, l1_valid = {l1_valid_now:.5f}',
                          flush=True)
                if save_path is not None:
                    if save_period <= 0 or (i_epoch + 1) % save_period == 0:
                        last_state = deepcopy(FullState(
                            model={k: v.cpu() for k, v in model_local.state_dict().items()},
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ))
                        torch.save(TrainResult(best_state=best_state, last_state=last_state,
                                               lambda_reg=lambda_reg, l0_train=np.asarray(l0_train),
                                               l1_train=np.asarray(l1_train),
                                               l0_valid=np.asarray(l0_valid),
                                               l1_valid=np.asarray(l1_valid))._asdict(),
                                   save_path + '.tmp') # saving as dict in case the def changes
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        os.rename(save_path + '.tmp', save_path)
            scheduler.step()

        last_state = deepcopy(FullState(
            model={k: v.cpu() for k, v in model_local.state_dict().items()},
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
        ))
        if return_best_epoch:
            model_local.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        train_result = TrainResult(best_state=best_state, last_state=last_state,
                                   lambda_reg=lambda_reg, l0_train=np.asarray(l0_train),
                                   l1_train=np.asarray(l1_train), l0_valid=np.asarray(l0_valid),
                                   l1_valid=np.asarray(l1_valid))

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if save_path is not None:
                torch.save(train_result._asdict(), save_path + '.tmp')
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(save_path + '.tmp', save_path)
            if verbose > 0:
                print(f'[{datetime.datetime.now()}]  finished training dim {model_local.i}, '
                      f'l0_valid_best = {np.asarray(l0_valid)[np.argmin(valid_loss)]:.5f}, '
                      f'l1_valid_best = {np.asarray(l1_valid)[np.argmin(valid_loss)]:.5f}',
                      flush=True)

        return train_result

    else:
        raise ValueError("I don't know how to train this quantile_net_1d.")


TrainResult = namedtuple('TrainResult', ['best_state', 'last_state', 'lambda_reg',
                                         'l0_train', 'l1_train', 'l0_valid', 'l1_valid'])


FullState = namedtuple('FullState', ['model', 'optimizer', 'scheduler'])


def _get_prev_state(save_path, device, verbose):
    if save_path is None or not os.path.isfile(save_path):
        if verbose > 0:
            print(f'[{datetime.datetime.now()}]  did not find previous checkpoint, will train from '
                  f'scratch.', flush=True)
        return None
    else:
        if verbose > 0:
            print(f'[{datetime.datetime.now()}]  found previous checkpoint, will resume training '
                  f'from that.', flush=True)
        return TrainResult(**torch.load(save_path, map_location=device))


def _decode_batch(batch_now, device):
    if isinstance(batch_now, torch.Tensor):
        return None, batch_now.to(device, torch.float)
    elif hasattr(batch_now, '__iter__') and len(batch_now) == 2:
        return batch_now[0].to(device, torch.float), batch_now[1].to(device, torch.float)
    else:
        raise ValueError


def _l0_lambda_l1(l0_valid, l1_valid, lambda_reg):
    return np.asarray(l0_valid) * (1 + lambda_reg * np.asarray(l1_valid))


def _check_convergence(l0_valid, l1_valid, lambda_reg, stop_after_epochs, stop_tol, max_epochs):
    loss = _l0_lambda_l1(l0_valid, l1_valid, lambda_reg)
    if len(loss) >= max_epochs:
        return True
    elif stop_after_epochs is None or len(loss) <= stop_after_epochs:
        return False
    else:
        return loss[-(stop_after_epochs + 1)] <= (1 + stop_tol) * np.nanmin(
            loss[-stop_after_epochs:])


# def _broadcast_convergence(stop_signal):
#     # Broadcast the stop signal to all processes
#     stop_tensor = torch.tensor(int(stop_signal), dtype=torch.int)
#     dist.broadcast(stop_tensor, src=0)
#     return stop_tensor.item() == 1


# def _check_convergence(l0_valid, l1_valid, lambda_reg, stop_after_epochs, stop_tol, max_epochs):
#     if (not dist.is_initialized()) or dist.get_rank() == 0:
#         loss = _get_full_loss(l0_valid, l1_valid, lambda_reg)
#         stop_signal = _local_convergence(loss, stop_after_epochs, stop_tol, max_epochs)
#     else:
#         stop_signal = False
#     return _broadcast_convergence(stop_signal) if dist.is_initialized() else stop_signal


def _compute_total_loss(local_l0, local_l1, local_n):
    if dist.is_initialized():
        dist.all_reduce(local_l0, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_l1, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    l0 = local_l0 / local_n
    l1 = local_l1 / local_n
    return l0.item(), l1.item()
