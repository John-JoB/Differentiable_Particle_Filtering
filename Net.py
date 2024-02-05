from typing import Callable
import torch as pt
from dpf.model import *
from numpy import sqrt
from dpf.utils import nd_select, normalise_log_quantity, batched_select

class FCNN(pt.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.Linear(in_dim, hidden_dim),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden_dim, hidden_dim),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze()


class RealNVP_Cond(pt.nn.Module):

    def __init__(self, dim, hidden_dim = 4, y_dim = 1, base_net=FCNN):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.t1 = base_net(self.dim_1 + y_dim, self.dim_2, hidden_dim)
        self.s1 = base_net(self.dim_1 + y_dim, self.dim_2, hidden_dim)
        self.t2 = base_net(self.dim_2 + y_dim, self.dim_1, hidden_dim)
        self.s2 = base_net(self.dim_2 + y_dim, self.dim_1, hidden_dim)

    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                # pass
                pt.nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, y):
        lower, upper = pt.concat((x[:, :, :self.dim_1], y), dim=-1), pt.concat((x[:, :, self.dim_1:], y), dim=-1)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * pt.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * pt.exp(s2_transformed)
        z = pt.cat([lower, upper], dim=2)
        log_det = pt.sum(s1_transformed, dim=2) + \
                  pt.sum(s2_transformed, dim=2)
        return z ,log_det
    
    def inverse(self, z, y):
        lower, upper = pt.concat((z[:,:,:self.dim_1], y), dim=-1), pt.concat((z[:,:,self.dim_1:], y), dim=-1)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * pt.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * pt.exp(-s1_transformed)
        x = pt.cat([lower, upper], dim=2)
        log_det = pt.sum(-s1_transformed, dim=2) + \
                  pt.sum(-s2_transformed, dim=2)
        return x, log_det
    

class Stacked_RealNVP_Cond(pt.nn.Module):
    def __init__(self, RealNVP_list) -> None:
        super().__init__()
        self.mods = pt.nn.ModuleList(RealNVP_list)

    def forward(self, x, y):
        Tx = x
        log_det = 0
        for mod in self.mods:
            Tx, log_det_t = mod.forward(Tx, y)
            log_det = log_det + log_det_t
        return Tx, log_det
    
    def inverse(self, z, y):
        Tz = z
        log_det = 0
        for mod in self.mods:
            Tz, log_det_t = mod.forward(Tz, y)
            log_det = log_det + log_det_t
        return Tz, log_det_t

class OneDFlow(pt.nn.Module):
    def __init__(self, steps = 10, y_dim = 2, hidden_dim = 4, lim = 4, scaling = None, loc  = None):
        super().__init__()
        self.steps=steps
        self.lim = lim
        self.scale_loc = True
        self.scaling = FCNN(y_dim, 1, hidden_dim)
        self.loc = FCNN(y_dim, 1, hidden_dim)
        self.knots = FCNN(y_dim, (steps-1)*3 + 1, hidden_dim*int(sqrt(steps)))
        self.sm = pt.nn.Softmax(dim=-1)
        self.sigm = pt.nn.Sigmoid()

    def forward(self, x, y):
        loc = self.loc(y)
        scale = pt.abs(self.scaling(y))
        knots = self.knots(y)
        if self.scale_loc:
            return x*scale + loc, pt.log(scale)
        widths = 2 * self.lim * self.sm(knots[..., :self.steps-1])
        heights = 2 * self.lim * self.sm(knots[..., self.steps-1:-self.steps])
        x_pos = pt.cumsum(widths, dim=-1) - self.lim
        z_pos = pt.cumsum(heights, dim=-1) - self.lim
        gradients = self.sigm(knots[..., -self.steps:])
        gradients[..., :-1] = 2*(widths/heights)*gradients[..., :-1]
        gradients[..., -1] = 2*(self.lim - z_pos[..., -2])/(self.lim - x_pos[..., -2])*gradients[..., -1]
        segments = pt.searchsorted(x_pos, x).detach()
        log_grad = pt.empty_like(x)
        z = pt.empty_like(x)
        for i in range(self.steps+1):
            if i == 0:
                mask = x < -self.lim
            else:
                mask = (segments == i-1)
    
            if i == 0 or i == self.steps:
                z[mask] = x[mask]
                log_grad[mask] = 0
                continue
            if i == 1:
                x_t = x + 4
            else:
                x_t = x - x_pos[..., i-2:i-1]
            b = gradients[..., i-1:i]
            a = (heights[..., i-1:i] - b*widths[..., i-1:i])/(widths[..., i-1:i]**2)
            if i==1:
                z[mask] = a[mask]*x_t[mask]**2 + b[mask]*x_t[mask] - 4
            else:
                z[mask] = a[mask]*x_t[mask]**2 + b[mask]*x_t[mask] + z_pos[..., i-2:i-1][mask]
            log_grad[mask] = pt.log((2*a[mask]*x_t[mask] + b[mask]))

        return z*scale + loc, log_grad + pt.log(scale)
    

    def inverse(self, z, y):
        loc = self.loc(y)
        scale = pt.abs(self.scaling(y))
        knots = self.knots(y)
        if self.scale_loc:
            return (z-loc)/(scale), -pt.log(scale)
        widths = 2 * self.lim * self.sm(knots[..., :self.steps-1])
        heights = 2 * self.lim * self.sm(knots[..., self.steps-1:-self.steps])
        x_pos = (pt.cumsum(widths, dim=-1) - self.lim)
        z_pos = (pt.cumsum(heights, dim=-1) - self.lim)
        pre_gradients = self.sigm(knots[..., -self.steps:])
        gradients = pt.empty_like(pre_gradients)
        gradients[..., :-1] = 2*(widths/heights)*pre_gradients[..., :-1]
        gradients[..., -1] = 2*(self.lim - z_pos[..., -2])/(self.lim - x_pos[..., -2])*pre_gradients[..., -1]
        z_scaled = (z - loc)/scale
        segments = pt.searchsorted(z_pos, z_scaled).detach()
        # print(segments[0, 2])
        # print(z_scaled[0, 2])
        # print(z_pos[0, 2])
        log_grad = pt.empty_like(z)
        x = pt.empty_like(z)
        
        for i in range(self.steps+1):
            if i == 0:
                mask = z_scaled < -self.lim
                segments[mask] = -1
            else:
                mask = (segments == i-1)
            if i == 0 or i == self.steps:
                x[mask] = z_scaled[mask]
                log_grad[mask] = 0
                continue
            if i == 1:
                z_t = z_scaled + 4
            else:
                z_t = z_scaled - z_pos[..., i-2:i-1]
            b = gradients[..., i-1:i]
            a = (heights[..., i-1:i] - b*widths[..., i-1:i])/(widths[..., i-1:i]**2)
            root_term = pt.sqrt(b[mask]**2 + 4*a[mask]*z_t[mask])
            # if i == 2:
            #     print(b[mask][0])
            #     print(a[mask][0])
            #     print(z_t[mask][0])
            #     print(widths[..., i-1:i][mask][0])
            #     print(root_term[0])
            x[mask] = -b[mask] - ((a.detach()[mask]>0)*2 - 1) * root_term
            log_grad[mask] = -pt.log(root_term)
            x[mask] = x[mask]/(2*a[mask])
            #print(pt.max(x[mask] - widths[..., i-1:i][mask]))
            if i == 1:
                x[mask] = x[mask] + 4
            else:
                x[mask] = x[mask] + x_pos[..., i-2:i-1][mask]
            #print(pt.max(x[mask]))
        #raise SystemExit(0)
        return x, log_grad - pt.log(scale)
            

class Markov_Switching(pt.nn.Module):
    def __init__(self, n_models:int, switching_diag: float, switching_diag_1: float, dyn = 'Boot', device:str ='cuda'):
        super().__init__()
        self.device=device
        self.dyn = dyn
        self.n_models = n_models
        tprobs = pt.ones(n_models) * ((1 - switching_diag - switching_diag_1)/(n_models - 2))
        tprobs[0] = switching_diag
        tprobs[1] = switching_diag_1
        self.switching_vec = pt.log(tprobs).to(device=device)
        self.dyn = dyn
        if dyn == 'Uni' or dyn == 'Deter':
            self.probs = pt.ones(n_models)/n_models
        else:
            self.probs = tprobs

    def init_state(self, batches, n_samples):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((batches, n_samples//self.n_models)).unsqueeze(2)
        return pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)

    def forward(self, x_t_1, t):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((x_t_1.size(0), x_t_1.size(1)//self.n_models)).unsqueeze(2) 
        shifts = pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1)])
        new_models = pt.remainder(shifts + x_t_1[:, :, 1], self.n_models)
        return new_models.unsqueeze(2)
    
    def get_log_probs(self, x_t, x_t_1):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts]


class Polya_Switching(pt.nn.Module):
    def __init__(self, n_models, dyn, device:str='cuda') -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        
    def init_state(self, batches, n_samples):
        self.scatter_v = pt.zeros((batches, n_samples, self.n_models), device=self.device)
        i_models = pt.multinomial(self.ones_vec, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        return pt.concat((i_models, pt.ones((batches, n_samples, self.n_models), device=self.device)), dim=2)

    def forward(self, x_t_1, t):
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:,:,1].unsqueeze(2).to(int), 1)
        c = x_t_1[:,:,2:] + self.scatter_v
        if self.dyn == 'Uni':
            return pt.concat((pt.multinomial(self.ones_vec,  x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1), 1]), c), dim=2)
        return pt.concat((pt.multinomial(c.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        probs = x_t[:, :, 2:]
        probs /= pt.sum(probs, dim=2, keepdim=True)
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        return pt.log(s_probs)

class NN_Switching(pt.nn.Module):

    def __init__(self, n_models, recurrent_length, device):
        super().__init__()
        self.device = device
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.self_forget = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid())
        self.scale = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_reccurrent = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Tanh(), pt.nn.Linear(recurrent_length, n_models))
        self.probs = pt.ones(n_models)/n_models

    def init_state(self, batches, n_samples):
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.r_length), device=self.device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 1].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=self.device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_recurrent = x_t_1[:, :, 2:]
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.scale(one_hot) * self.to_reccurrent(one_hot)
        return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:,:,1].to(int)
        probs = pt.abs(self.output_layer(x_t[:, :, 2:]))
        probs = probs / pt.sum(probs, dim=2, keepdim=True)
        log_probs = batched_select(probs, models)
        return pt.log(log_probs+1e-7)
    
class NN_Switching_LSTM(pt.nn.Module):

    def __init__(self, n_models, recurrent_length, device:str = 'cuda'):
        super().__init__()
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_hidden = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Tanh())
        self.self_gate = pt.nn.Sequential(pt.nn.Linear(2*n_models, recurrent_length), pt.nn.Sigmoid())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(2*n_models, n_models), pt.nn.Sigmoid())
        self.tanh = pt.nn.Sigmoid()
        self.probs = pt.ones(n_models)/n_models
        self.t_probs = pt.zeros(n_models, device=self.device)

    def init_state(self, batches, n_samples):
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.n_models + self.r_length), device=self.device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 1].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=self.device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_probs = x_t_1[:, :, 2:2+self.n_models]
        old_recurrent = x_t_1[:, :, 2+self.n_models:]
        in_vector = pt.concat((one_hot, old_probs), dim=2)
        recurrent = old_recurrent * self.forget(in_vector)
        recurrent += self.self_gate(in_vector)*self.to_hidden(in_vector)
        new_probs = self.output_layer(in_vector) * self.tanh(recurrent)
        new_probs = new_probs / pt.sum(new_probs, dim=2, keepdim=True)
        self.t_probs = new_probs
        return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), new_probs, recurrent), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:,:,1].to(int)
        log_probs = batched_select(self.t_probs, models)
        return pt.log(log_probs)
    
class Simple_NN(pt.nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = pt.nn.Sequential(pt.nn.Linear(input, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, output))

    def forward(self, in_vec):
        return self.net(in_vec.unsqueeze(1)).squeeze()

class PF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, dyn ='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device = device)
        self.b = pt.tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.var_factor = -1/(2*var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)
                                      
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
        new_models = self.switching_dyn(x_t_1, t)
        index = new_models[:,:,0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        locs = (scaling*pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias)

        return self.var_factor * ((self.y[t] - locs)**2)
    
    def observation_generation(self, x_t):
        noise = self.y_dist.sample((x_t.size(0), 1)).to(device=self.device)
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * pt.sqrt(pt.abs(x_t[:, :, 0])) + bias).unsqueeze(2) + noise)
        return new_pos


class RSDBPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        new_models = self.switching_dyn(x_t_1, t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=self.device)
        index = new_models[:, :, 0].to(int)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)
    
'''

class RSDBPF_Guided_fake(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.prop_models = pt.nn.ModuleList([FCNN(2, 8, 1) for _ in range(n_models)])
        self.sd_prop = pt.nn.ModuleList([Simple_NN(2, 8, 1) for _ in range(n_models)])
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.guided = True
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device).squeeze()
        new_models = self.switching_dyn(x_t_1, t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=device)
        index = new_models[:, :, 0].to(int)
        new_pos = pt.empty_like(locs)
        sd = pt.empty_like(locs)
        expanded_ys = self.y[t].unsqueeze(1).expand(-1, x_t_1.size(1), -1)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
            conditioning = pt.concat((locs[mask].unsqueeze(1), expanded_ys[mask]), dim=1)
            sd[mask] = pt.abs(self.sd_prop[m](conditioning))
            new_pos[mask] = self.prop_models[m](conditioning)
        new_pos = new_pos + noise*sd
        self.prop_densities = (-1/2)*(noise**2) - pt.log(sd+1e-5)
        self.dyn_densities = (-1/(2*self.sd_d**2))*(locs - new_pos)**2
        self.Ratio = self.dyn_densities - self.prop_densities
        new_pos = new_pos.unsqueeze(2)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1) + self.Ratio

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)

class RSDBPF_Guided(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.prop_models = pt.nn.ModuleList([OneDFlow(10, 2, 4, 4) for _ in range(n_models)])
        self.guided = False
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device) 
        new_models = self.switching_dyn(x_t_1, t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=device)
        index = new_models[:, :, 0].to(int)
        if self.guided:
            new_pos = pt.empty_like(locs)
            log_det = pt.empty_like(locs)
            expanded_ys = self.y[t].unsqueeze(1).expand(-1, x_t_1.size(1), -1)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
            if self.guided:
                conditioning = pt.concat((locs[mask].unsqueeze(1), expanded_ys[mask]), dim=1)
                a, b = self.prop_models[m](noise[mask], conditioning)
                new_pos[mask], log_det[mask] = a.squeeze(), b.squeeze()
        if self.guided:
            self.prop_densities = (-1/2)*(noise.squeeze()**2) - log_det
            self.dyn_densities = (-1/(2*self.sd_d**2))*(locs - new_pos)**2
            self.Ratio = self.dyn_densities - self.prop_densities
            new_pos = new_pos.unsqueeze(2)
        else:
            new_pos = (locs.unsqueeze(2) + noise*self.sd_d)
            self.Ratio = pt.zeros_like(new_pos).squeeze(dim=2)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1) + self.Ratio

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)
    

class RSDBPF_2(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:pt.device = device):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, 8, 1) for _ in range(n_models)])
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).tile((1, self.n_models)).unsqueeze(2)
        init_regimes = pt.repeat_interleave(pt.arange(self.n_models, device=device), n_samples).repeat((batches, 1))
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        init_regimes = pt.tile(init_regimes, (1, self.n_models, 1))
        for m in range(self.n_models):
            init_regimes[:, m*n_samples:(m+1)*n_samples, 0] = m
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        N = x_t_1.size(1)
        noise = self.x_dist.sample([x_t_1.size(0), N*self.n_models]).to(device=device) * self.sd_d
        new_models = self.switching_dyn(x_t_1, t)
        new_models = pt.tile(new_models, (1, self.n_models, 1))
        locs = pt.empty((x_t_1.size(0), N*self.n_models), device=device)
        for m in range(self.n_models):
            new_models[:, m*N:(m+1)*N, 0] = m
            locs[:, m*N:(m+1)*N] = self.dyn_models[m](x_t_1[:,:,0].unsqueeze(2))
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1.tile((1,self.n_models,1)))

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)
    

class DBPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, switching_diag:float, switching_diag_1:float, device:str = 'cuda'):
        super().__init__(self.device)
        self.n_models = n_models
        self.dyn_model = Simple_NN(1, 8, 1)
        self.obs_model = Simple_NN(1, 8, 1)
        self.sd_d = pt.nn.Parameter(pt.rand(1, device=self.device)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1, device=self.device)*0.4 + 0.1)
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.PF_type = 'Bootstrap'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        return init_locs               
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        locs = self.dyn_model(x_t_1)
        new_pos = locs.unsqueeze(2) + noise
        return new_pos
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return nd_select(self.switching_vec, shifts)

    def log_f_t(self, x_t, t: int):
        locs = self.obs_model(x_t)
        return self.var_factor * (self.y[t] - locs)**2
    
class RSDBPF_cheat(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, var_s:float, switching_dyn:pt.nn.Module, switching_diag:float, switching_diag_1:float, device:'str' = device):
        super().__init__(device)
        self.a = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.b = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.c = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.d = pt.nn.Parameter(pt.rand(n_models, requires_grad=True)-0.5)
        self.sd_d = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1, device=device)*0.4 + 0.1)
        self.n_models = n_models
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        
        self.switching_vec = pt.ones(self.n_models) * ((1 - switching_diag - switching_diag_1)/(self.n_models - 2))
        self.switching_vec[0] = switching_diag
        self.switching_vec[1] = switching_diag_1
        self.switching_vec = pt.log(self.switching_vec).to(device=device)
        self.PF_type = 'Guided'

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*self.sd_o**2)
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=device).unsqueeze(2)
        init_regimes = pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=device)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=device)*self.sd_d
        index = self.switching_dyn(x_t_1, t).to(int)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.a[m]*(x_t_1[:,:,0][mask]) + self.b[m]
        new_pos = (locs.unsqueeze(2) + noise)
        test = new_pos.isnan()
        if pt.any(test):
            print('Error nan')
            raise SystemExit(0)
        return pt.cat((new_pos, index.unsqueeze(2)), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return nd_select(self.switching_vec, shifts)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=device)
        pos = pt.abs(x_t[:,:,0])
        #Needed for numerical stability of backward pass
        pos = pt.where(pos < 1e-5, pos + 1e-5, pos)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.c[m]*pt.sqrt(pos[mask]) + self.d[m]
        return self.var_factor * (self.y[t] - locs)**2
    
class Generates_0(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, device:pt.device = device):
        super().__init__(device)
        self.PF_type = 'Bootstrap'

    def M_0_proposal(self, batches:int, n_samples: int):
        return pt.zeros((batches, n_samples, 1), device=device)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        return pt.zeros_like((x_t_1))
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0, n_samples: int):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=device)

    def log_R_t(self, x_t, x_t_1, t: int):
        None

    def log_f_t(self, x_t, t: int):
        return pt.zeros((x_t.size(0), x_t.size(1)), device=device)
        
class LSTM(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, n_layers) -> None:
        super().__init__()
        self.lstm = pt.nn.LSTM(obs_dim, hid_dim, n_layers, True, True, 0.0, False, state_dim, device)

    def forward(self, y_t):
            return self.lstm(y_t)[0]
            '''