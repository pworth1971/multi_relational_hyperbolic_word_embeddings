import numpy as np
import torch

import platform

if platform.system() == 'Linux':
    # Import Linux-specific modules
    import pycuda.driver as cuda

from torch import Tensor
from definitions_learning.models.multi_relational.utils import *
#from web.evaluate import poincare_distance


# ----------------------------------------------------------------------------------
# pjw updates:
# code to support Apple silicon which includes the updated device 
# type and the conversion from float64 (double) to float32
#

#
# The MuRP (Multi-relational Poincaré) class is a PyTorch neural network 
# model designed for multi-relational learning tasks, specifically embedding 
# entities and relations in a hyperbolic space (Poincaré ball model). This model 
# is used to represent hierarchical relationships in a more compact space 
# compared to Euclidean embeddings.
#
# modificatins to accept device (CPU, MPS or CUDA) as a parameter to
# class upon instantiation, for both classes, affecting all methods
#
# -----------------------------------------------------------------------------------
class MuRP(torch.nn.Module):
    def __init__(self, d, dim):

        super(MuRP, self).__init__()
        
        if platform.system() == 'Linux':
            print("Setting up GPU parallelism...")
            cuda.init()
            print("Number of GPUs:", cuda.Device.count())

        # set runtime device (Chip Set)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print("MuRP::device:", self.device)
        print()

        #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #device = torch.device(device)           # set runtime device (Chip Set)
    
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0).to(self.device)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.float32, device=self.device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0).to(self.device)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.float32, device=self.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True, device=self.device))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=self.device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=self.device))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        
        #print("MuRP::forward(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
        device = torch.device(self.device)           # set runtime device (Chip Set)  
        
        u = self.Eh.weight[u_idx].to(device)
        v = self.Eh.weight[v_idx].to(device)
        Ru = self.Wu[r_idx].to(device)
        rvh = self.rvh.weight[r_idx].to(device)

        eps = 1e-5
        u_nrm = torch.linalg.vector_norm(u, 2, dim=-1, keepdim=True)
        v_nrm = torch.linalg.vector_norm(v, 2, dim=-1, keepdim=True)
        rvh_nrm = torch.linalg.vector_norm(rvh, 2, dim=-1, keepdim=True)
        u_nrm_msk = u_nrm >= 1
        v_nrm_msk = v_nrm >= 1
        rvh_nrm_msk = rvh_nrm >= 1
        u_nrm_e = u_nrm - eps
        v_nrm_e = v_nrm - eps
        rvh_nrm_e = rvh_nrm - eps

        u = (u/u_nrm_e) * u_nrm_msk + u * torch.logical_not(u_nrm_msk)
        v = (v/v_nrm_e) * v_nrm_msk + v * torch.logical_not(v_nrm_msk)
        rvh = (rvh/rvh_nrm_e) * rvh_nrm_msk + rvh * torch.logical_not(rvh_nrm_msk)

        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)

        um_nrm = torch.linalg.vector_norm(u_m, 2, dim=-1, keepdim=True)
        vm_nrm = torch.linalg.vector_norm(v_m, 2, dim=-1, keepdim=True)
        um_nrm_msk = um_nrm >= 1
        vm_nrm_msk = vm_nrm >= 1
        um_nrm_e = um_nrm - eps
        vm_nrm_e = vm_nrm - eps
        u_m = (u_m / um_nrm_e) * um_nrm_msk + u_m * torch.logical_not(um_nrm_msk)
        v_m = (v_m / vm_nrm_e) * vm_nrm_msk + v_m * torch.logical_not(vm_nrm_msk)

        sqdist = (2.*artanh(torch.clamp(torch.linalg.vector_norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx]

    def one_shot_encoding(self, v_idx, r_idx):
        
        print("MuRP::one_shot_encoding(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        device = torch.device(self.device)           # set runtme device (Chip Set)
        
        v = self.Eh.weight[v_idx].to(device)
        Ru = self.Wu[r_idx].to(device)
        rvh = self.rvh.weight[r_idx].to(device)

        eps = 1e-5
        v_nrm = torch.linalg.vector_norm(v, 2, dim=-1, keepdim=True)
        rvh_nrm = torch.linalg.vector_norm(rvh, 2, dim=-1, keepdim=True)
        v_nrm_msk = v_nrm >= 1
        rvh_nrm_msk = rvh_nrm >= 1
        v_nrm_e = v_nrm - eps
        rvh_nrm_e = rvh_nrm - eps

        v = (v/v_nrm_e) * v_nrm_msk + v * torch.logical_not(v_nrm_msk)
        rvh = (rvh/rvh_nrm_e) * rvh_nrm_msk + rvh * torch.logical_not(rvh_nrm_msk)

        v_m = p_sum(torch.mean(v,1), torch.mean(rvh,1))

        return v_m

    def one_shot_encoding_avg(self, v_idx):
        
        print("MuRP::one_shot_encoding_avg(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        device = torch.device(self.device)
        
        v = self.Eh.weight[v_idx].to(device)

        return torch.mean(v,1)
    

#
# also updated for MPS (Apple silicon) support
#
# The MuRE class is a PyTorch neural network model designed for multi-relational 
# learning tasks, specifically for embedding entities and relations in a Euclidean 
# space. Here's a detailed breakdown of what the MuRE class does:
# 
# The MuRE model maps entities and relations to a Euclidean space and computes 
# scores for triples (subject, relation, object) based on their embeddings. It uses these 
# scores to learn and evaluate relationships in a knowledge graph or similar 
# relational data structure.
#   
class MuRE(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRE, self).__init__()
        
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #device = torch.device(device)           # set runtime device (Chip Set)
        
        # set runtime device (Chip Set)
        if torch.backends.cuda.is_built():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print("MuRE::device:", self.device)

        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0).to(self.device)
        self.E.weight.data = self.E.weight.data.float()
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.float32, device=self.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True, device=self.device))
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0).to(self.device)
        self.rv.weight.data = self.rv.weight.data.float()
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.float32, device=self.device))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=self.device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=self.device))
        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        
        print("MuRE::forward(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        device = torch.device(self.device)                   # set runtime device (Chip Set)
        
        u = self.E.weight[u_idx].to(device)
        v = self.E.weight[v_idx].to(device)
        Ru = self.Wu[r_idx].to(device)
        rv = self.rv.weight[r_idx].to(device)
        
        u_W = u * Ru

        sqdist = torch.sum(torch.pow(u_W - (v + rv), 2), dim=-1)
        return -sqdist + self.bs[u_idx] + self.bo[v_idx]

    def one_shot_encoding(self, v_idx, r_idx):
        
        print("MuRP::one_shot_encoding(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        device = torch.device(self.device)                   # set runtime device (Chip Set)
        
        v = self.E.weight[v_idx].to(device)
        Ru = self.Wu[r_idx].to(device)
        rv = self.rv.weight[r_idx].to(device)
        
        u_W = torch.mean(v, 1) + torch.mean(rv, 1)
        
        return u_W

    def one_shot_encoding_avg(self, v_idx):
        
        print("MuRP::one_shot_encoding_avg(): self.device: ", self.device)

        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        device = torch.device(self.device)              # set runtime device (Chip Set)
        
        v = self.E.weight[v_idx].to(device)
        
        return torch.mean(v, 1)                         # return the average v vector from all the terms in the definition
