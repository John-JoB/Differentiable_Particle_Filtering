import torch as pt
from torch import nn
from .simulation import Differentiable_Particle_Filter
from copy import copy
from tqdm import tqdm
from typing import Callable, Iterable
from matplotlib import pyplot as plt
import numpy as np
from .results import Reporter
from torch.utils.tensorboard import SummaryWriter
import time
import torch.autograd.profiler as profiler
import warnings
from copy import deepcopy
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
from .loss import Loss


def _test(
        DPF: Differentiable_Particle_Filter, 
        loss: Loss, 
        T: int, 
        data: pt.utils.data.DataLoader, 
        ):
    DPF.eval()
    with pt.inference_mode():
        for i, simulated_object in enumerate(tqdm(data)):
            loss.clear_data()
            DPF(simulated_object, T, loss.get_reporters())
            loss_t = loss.forward()
            try:
                loss = pt.concat(loss, loss_t.cpu().detach().numpy())
            except:
                loss = loss_t.cpu().detach().numpy()   
    print(f'Test loss: {np.mean(loss)}')
    return loss

def e2e_train(
        DPF: Differentiable_Particle_Filter,
        opt: pt.optim.Optimizer,
        loss: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        test_scaling: float=1,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True
        ):
    train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
    if batch_size[0] == -1:
        batch_size[0] == len(train_set)
    if batch_size[1] == -1:
        batch_size[1] == len(valid_set)
    if batch_size[2] == -1:
        batch_size[2] == len(test_set)

    train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data.collate, num_workers= data.workers)
    valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
    test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    
    for epoch in range(epochs):
        DPF.train()
        if verbose:
            train_it = enumerate(tqdm(train))
        else:
            train_it = enumerate(train)
        for b, simulated_object in train_it:
            opt.zero_grad()
            loss.clear_data()
            loss.register_data(simulated_object)
            DPF(simulated_object, T, loss.get_reporters())
            loss.forward()
            loss.backward()
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()
        opt_schedule.step()
        DPF.eval()
        with pt.inference_mode():
            for simulated_object in valid:
                loss.clear_data()
                DPF(simulated_object, T, loss.get_reporters())
                test_loss[epoch] += loss.forward().item()
            test_loss[epoch] /= len(valid)

        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(DPF.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(np.sqrt(train_loss[epoch*len(train):(epoch+1)*len(train)]))}')
            print(f'Validation loss: {np.sqrt(test_loss[epoch])}\n')


    if verbose:
        plt.plot(train_loss)
        plt.plot((np.arange(len(test_loss)) + 1)*(len(train_loss)/len(test_loss)),test_loss)
        plt.show()
    DPF.load_state_dict(best_dict)
    DPF.n_particles *= test_scaling
    DPF.ESS_threshold *= test_scaling
    return _test(DPF, loss, T, test)

def test(DPF: Differentiable_Particle_Filter,
        loss: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size:int, 
        fraction:float, 
        ):
        test_set, _ =  pt.utils.data.random_split(data, [fraction, 1-fraction])
        if batch_size == -1:
            batch_size == len(test_set)
        test = pt.utils.data.DataLoader(test_set, min(batch_size, len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
        _test(DPF, loss, T, test)
