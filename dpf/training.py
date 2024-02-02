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


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                try:
                    assert fn in fn_dict, fn
                except:
                    return
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

def train_optimal_proposals(DPF, data, lr=0.001, epochs=10):
    proposals = DPF.model.prop_models
    likelihoods = DPF.model.obs_models
    dynamics = DPF.model.dyn_models
    dynamic_sigma = DPF.model.sd_d
    likelihood_sigma = DPF.model.sd_o
    

    def evaluate_log_gausian_density(input, mean, sigma):
        prefactor =  -1/(2*sigma**2)
        squared_error = (input - mean)**2
        return prefactor*squared_error 
    
    for m in range(len(proposals)):
        opt = pt.optim.AdamW(proposals[m].parameters(), lr=lr)
        print(f'===================================')
        print(f'Training optimal proposal for model {m+1}/{len(proposals)}')
        print(f'-----------------------------------\n')
        for e in range(epochs):
            #if e == 5:
                #proposals[m].scale_loc = False
            mean_divergence = 0
            for b, object in enumerate(data):
                opt.zero_grad()
                xs = object.state[:, 1:, 0:1]
                x_1s = object.state[:,:-1, 0:1]
                ys = object.observations[:, 1:, 0:1]
                ds = dynamics[m](x_1s)
                ls = likelihoods[m](xs)
                prop_cond = pt.concat((ds.unsqueeze(-1), ys), dim=-1)
                dynamic_density = evaluate_log_gausian_density(xs.squeeze(), ds, dynamic_sigma)
                likelihood_density = evaluate_log_gausian_density(ys.squeeze(), ls, likelihood_sigma)
                normal_prop, log_det= proposals[m].inverse(xs, prop_cond)
                proposal_density = log_det + evaluate_log_gausian_density(normal_prop, 0, 1)
                target_density = dynamic_density + likelihood_density
                target_density -= pt.max(target_density.detach())
                target_density.detach()
                divergence = -pt.sum(pt.exp(target_density)*proposal_density.squeeze())
                #print(r.grad)
                #raise SystemExit(0)
                mean_divergence += divergence.item()
                opt.step()
            if (e+1)%5 == 0:
                print(f'Epoch {e}:')
                print(f'Divergence: {mean_divergence/len(data)}')
            

def test(
        DPF: Differentiable_Particle_Filter, 
        loss_fn: Loss, 
        statistics: Iterable[Reporter], 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int,
        ):
    DPF.eval()
    try:
        test_loader = pt.utils.data.DataLoader(data, batch_size, shuffle=True, collate_fn=data.collate)
    except:
        test_loader = data
    out_stat = []
    DPF.n_particles = 2000
    DPF.ESS_threshold = 2001
    with pt.inference_mode():
        for i, simulated_object in enumerate(tqdm(test_loader)):
            statistics_ = [copy(statistic) for statistic in statistics]
            DPF(simulated_object, T, statistics_)
            loss_t = pt.mean(loss_fn.per_point_loss(statistics_[0], simulated_object), dim=1)
            try:
                loss = pt.concat(loss, pt.sqrt(loss_t).cpu().detach().numpy())
            except:
                loss = pt.sqrt(loss_t).cpu().detach().numpy()
            #if i == 0:
                #plt.plot(statistics_[0].results[0, :20, 0].detach().cpu().numpy(), statistics_[0].results[0, :20, 1].detach().cpu().numpy())
                #plt.plot(simulated_object.state[0, :20, 0].detach().cpu().numpy(), simulated_object.state[0, :20, 1].detach().cpu().numpy())
                #plt.xlim((0, 1000))
                #plt.ylim((0, 500))
                #plt.show()
                #DPF.initialise(simulated_object)
                #DPF.display_particles(50, [0,1], ('a', ''), 'd')
            
        
    print(f'Mean rmse: {np.mean(loss)}')
    print(f'Worst rmse: {np.max(loss)}')
    print(f'Best rmse: {np.min(loss)}')
    print(f'Std rmse: {np.std(loss)}')
    return loss

def e2e_train(
        DPF: Differentiable_Particle_Filter, 
        loss_fn: Loss, 
        statistics: Iterable[Reporter], 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int, 
        train_fraction: float, 
        epochs: int, 
        lr: float,
        p_scaling: float,
        verbose = True
        ):
    train_set, test_set, f_test_set = pt.utils.data.random_split(data, [0.5, 0.25, 0.25])
    train = pt.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=data.collate, num_workers= data.workers)
    valid = pt.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
    f_test = pt.utils.data.DataLoader(f_test_set, len(f_test_set), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    try:
        opt = pt.optim.SGD(DPF.parameters(), lr=lr, momentum=0.9)
        #opt = pt.optim.AdamW(DPF.parameters(), lr=lr)
    except:
        return test(DPF, loss_fn, statistics, T, f_test, len(f_test))
    min_valid_loss = pt.inf
    #opt_schedule = pt.optim.lr_scheduler.OneCycleLR(opt, max_lr= lr, steps_per_epoch=len(train), pct_start=0.2, epochs=epochs)
    opt_schedule = pt.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20, 30, 40, 50], gamma=0.5)
    
    for epoch in range(epochs):
        DPF.train()
        if verbose:
            train_it = enumerate(tqdm(train))
        else:
            train_it = enumerate(train)
        if epoch%10 == 0 & epoch != 0:
            DPF.n_particles *= p_scaling
            DPF.ESS_threshold *= p_scaling
        for b, simulated_object in train_it:
            #with pt.autograd.set_detect_anomaly(True):
            opt.zero_grad()
            statistics_ = [copy(statistic) for statistic in statistics]

            DPF(simulated_object, T, statistics_)
            loss = loss_fn.forward(statistics_[0], simulated_object)
        #print(loss.item())
            loss.backward()
            opt.step()
            if pt.any(pt.logical_or(pt.isnan(loss), pt.isinf(loss))):
                DPF.load_state_dict(best_dict)
                print('NaN encountered, reverting to stable state')
            
            train_loss[b + len(train)*epoch] = loss.item()
        #print(np.sqrt(loss.item()))
        opt_schedule.step()
        DPF.eval()
        with pt.inference_mode():
            for simulated_object in valid:
                statistics_ = [copy(statistic) for statistic in statistics]
                DPF(simulated_object, T, statistics_)
                test_loss[epoch] += loss_fn.forward(statistics_[0], simulated_object).item()
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
    test(DPF, loss_fn, statistics, T, f_test, 100)
    return test(DPF, loss_fn, statistics, T, f_test, 100)

def test_NN(
        NN: pt.nn.Module, 
        loss_fn: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int, 
        ):
    
    NN.to(device=device)
    NN.eval()
    with pt.inference_mode():
        try:
            test_loader = pt.utils.data.DataLoader(data, batch_size, shuffle=True, collate_fn=data.collate)
        except:
            test_loader = data
        loss = 0
        for i, sim_object in enumerate(tqdm(test_loader)):
            data_tensor = sim_object.observations[:, :T+1, :]
            output = NN(data_tensor)
            loss = pt.mean(loss_fn.per_point_loss(output, sim_object), dim=1)
            loss = pt.sqrt(loss).cpu().detach().numpy()
            print(f'Mean rmse: {np.mean(loss)}')
            print(f'Worst rmse: {np.max(loss)}')
            print(f'Best rmse: {np.min(loss)}')
            print(f'Std rmse: {np.std(loss)}')
            print(f'Loss: {loss/len(test_loader)}')
        return loss

def train_nn(NN: pt.nn.Module, loss_fn: Loss, T: int, data: pt.utils.data.Dataset, batch_size:int, train_fraction:float, epochs:int, lr: float, verbose: bool = True):
    NN.to(device=device)
    train_set, test_set, f_test_set = pt.utils.data.random_split(data, [train_fraction, (1 - train_fraction)/2., (1 - train_fraction)/2.])
    train = pt.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=data.collate)
    valid = pt.utils.data.DataLoader(test_set, len(test_set), shuffle=True, collate_fn=data.collate)
    f_test = pt.utils.data.DataLoader(f_test_set, len(f_test_set), collate_fn=data.collate)
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    opt = pt.optim.SGD(NN.parameters(), lr=lr, momentum=0.9)
    opt_schedule = pt.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20, 30, 40, 50], gamma=0.5)
    min_valid_loss = pt.inf
    
    for epoch in range(epochs):
        NN.train()
        if verbose:
            train_it = enumerate(tqdm(train))
        else:
            train_it = enumerate(train)
        for b, sim_object in train_it:
            data_tensor = sim_object.observations[:, :T+1, :]
            output = NN(data_tensor)
            loss = loss_fn.forward(output, sim_object)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()
        opt_schedule.step()
        NN.eval()
        with pt.inference_mode():
            for sim_object in valid:
                data_tensor = sim_object.observations[:, :T+1, :]
                output = NN(data_tensor)
                loss = loss_fn.forward(output, sim_object)
                test_loss[epoch] += loss.item()
            test_loss[epoch] /= len(valid)

        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(NN.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')

    if verbose:
        plt.plot(train_loss)
        plt.plot((np.arange(len(test_loss)) + 1)*(len(train_loss)/len(test_loss)),test_loss)
        plt.show()
    NN.load_state_dict(best_dict)
    return test_NN(NN, loss_fn, T, f_test, len(f_test))
    
    




def grid_search(function: Callable, param_dict: dict[Iterable]):
    assert set(function.__code__.co_varnames[:function.__code__.co_argcount]) == set(param_dict.keys())
    min_loss = None
    min_set = None
    def item_gen():
        nonlocal param_dict
        param_list = param_dict.items()
        counts = np.ones(len(param_list)+1, dtype=int)
        quants = np.array([len(i[1]) for i in param_list])
        counts[1:] = np.cumprod(quants)
        for i in range(counts[-1]):
            yield {k: v[(i//counts[j])%len(v)] for j,(k,v) in enumerate(param_list)} 

    for param_set in item_gen():
        print(param_set)
        try:
            loss = function(**param_set)
            print(loss)
            print('\n')
            if min_loss is None or loss < min_loss:
                min_loss = loss
                min_set = param_set
        except AssertionError as e:
            print(e)
            print('Failed')
            print('\n')
        
    print('\n\n')
    print('-----------------------------------------')
    print(f'Minimum loss: {min_loss}')
    print('With parameters:')
    print(min_set)


def e2e_train_jiaxi(
        DPF: Differentiable_Particle_Filter, 
        loss_fn: Callable[[Iterable[Reporter]], pt.Tensor], 
        statistics: Iterable[Reporter], 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: int, 
        test_num: float, 
        epochs: int, 
        lr: float,
        clip: float
        ):
    
    DPF.to(device=device)
    train_set, test_set = pt.utils.data.random_split(data, [len(data) - test_num, test_num])
    train = pt.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=data.collate)
    test = pt.utils.data.DataLoader(test_set, len(test_set), shuffle=True, collate_fn=data.collate)
    train_loss = np.zeros(len(train))
    test_loss = np.zeros(epochs)
    opt = pt.optim.AdamW(DPF.parameters(), lr)
    DPF.train()
    for epoch in range(epochs):
        for b, simulated_object in enumerate(train):
            opt.zero_grad()
            statistics_ = [copy(statistic) for statistic in statistics]
            if b != 10:
                DPF(simulated_object, T, statistics_)
            else:
                with profiler.profile(with_stack=True, with_modules=True, use_cuda=True) as prof:
                    DPF(simulated_object, T, statistics_)
            loss = loss_fn(statistics_, simulated_object)
            loss.backward()
            opt.step()
            if b % 10 == 0:
                ps = DPF.parameters()
                print(f'Iteration{b}: loss, {loss.item()}; theta 1, {ps.__next__().item()}, theta 2, {ps.__next__().item()}')
                
        for simulated_object in test:
            statistics_ = [copy(statistic) for statistic in statistics]
            DPF(simulated_object, T, statistics_)
            test_loss[epoch] += loss_fn(statistics_, simulated_object).item()
        test_loss[epoch] /= len(test)

        print(f'Epoch {epoch}:')
        print(f'Test loss: {test_loss[epoch]}\n')
    
    plt.plot(train_loss)
    #plt.plot(test_loss)
    plt.show()
        
        
