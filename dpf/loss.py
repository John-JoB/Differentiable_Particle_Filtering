import torch as pt
from torch._tensor import Tensor
from .results import Reporter
from .model import Observation_Queue
from typing import Any, Callable
from abc import ABCMeta
import time
 
class Loss(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.t = [0]

    def per_point_loss(self, *args) -> pt.Tensor:
        pass

    def forward(self, *args) -> pt.Tensor:
        return pt.mean(self.per_point_loss(*args))

    def __call__(self, *args):
        return self.forward(*args)

class Supervised_L2_Loss(Loss):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def per_point_loss(self, prediction:Reporter, truth:Observation_Queue) -> pt.Tensor:
        try:
            results = self.function(prediction.results)
        except AttributeError:
            results = prediction
        g_truth = self.function(truth.state)[:, :results.size(1), :]
        return pt.sum((results - g_truth)**2, dim=2)
    

class Maze_Supervised_Loss(Loss):
    def __init__(self, maze_no):
        super().__init__()
        if maze_no == 1:
            self.maze_size = (10, 5)
        elif maze_no == 2:
            self.maze_size = (15, 9)
        elif maze_no == 3:
            self.maze_size = (20, 13)
        else:
            raise ValueError('maze_no should be between 1 and 3')
    
    def per_point_loss(self, prediction:Reporter, truth:Observation_Queue) -> pt.Tensor:
        results = prediction.results
        g_truth = truth.state[:, :results.size(1), :]
        a_diff = pt.abs(results[:,:,2:3] - g_truth[:,:,2:3])
        diff = pt.concat(((results[:,:,0:1] - g_truth[:,:,0:1])/(self.maze_size[0]*50), 
                         (results[:,:,1:2] - g_truth[:,:,1:2])/(self.maze_size[0]*50), 
                         (pt.min(a_diff, 2*pt.pi - a_diff))/(2*pt.pi)), dim=2)
        return pt.sum((diff)**2, dim=2)


    
class Magnitude_Loss(Loss):
    def __init__(self, function, sign):
        super().__init__()
        self.function = function
        self.sign = sign

    def per_point_loss(self, reporter) -> pt.Tensor:
        return self.function(reporter.results) * self.sign
    
class Encoder_Loss(Loss):
    def __init__(self, encoder, decoder, include_pos: False, state_encoder: None, loss_ratio = 1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_ratio = loss_ratio
        if include_pos:
            self.state_encoder = state_encoder
            
    
    def per_point_loss(self, truth: Observation_Queue, trajectory_length: int) -> Tensor:
        obs = pt.concat((truth.observations[0], truth.observations[1]), dim=2)
        obs = obs[:, :trajectory_length].flatten(0, 1)
        encoded, obs = self.encoder(obs, True)
        decoded = self.decoder(encoded)
        if self.loss_ratio !=1:
            encoded_state = self.state_encoder(truth.state[:,:trajectory_length,:]).flatten(0,1)
            return self.loss_ratio * pt.mean((decoded - obs)**2, dim = (1,2,3)) + (1-self.loss_ratio)*pt.mean((encoded - encoded_state)**2, dim=(1))
        return pt.sum((decoded - obs)**2, dim = (1,2,3)) 
        

        

class Masked_Loss(Loss):
    def __init__(self, loss:Loss):
        super().__init__()
        self.loss = loss

    def per_point_loss(self, mask, *args) -> pt.Tensor:
        return (self.loss.per_point_loss(*args)*mask) * ((mask.size(0)*mask.size(1))/ pt.sum(mask).item())
    
class Compound_Loss(Loss):

    def __init__(self, loss_list):
        super().__init__()
        self.loss_list = loss_list

    def per_point_loss(self, arg_list) -> pt.Tensor:
        for i, (l, a) in enumerate(zip(self.loss_list, arg_list)):
            if i == 0:
                loss = l[1] * l[0].per_point_loss(*a)
            else:
                loss = loss + l[1] * l[0].per_point_loss(*a)
        return loss