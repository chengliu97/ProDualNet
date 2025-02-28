# This function is based on ProteinMPNN/Pifold/esm, under the MIT License.
# Source: https://github.com/dauparas/ProteinMPNN, https://github.com/A4Bio/PiFold,https://github.com/facebookresearch/esm
from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from transformer import TransformerBlock

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def cal_dihedral(X, eps=1e-7):
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:,2:,:] # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)
    
    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    
    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v* u_1).sum(-1)) * torch.acos(cosD) # TODO: sign
    
    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    # psi, omega, phi
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) # ['N', 'CA', 'C', 'O']
    D = cal_dihedral(X)
    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3)) 
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    # alpha, beta, gamma
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ...
    cosD = (u_0*u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features

def _hbonds(X, E_idx, mask_neighbors, eps=1E-3):
    X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

    X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
    X_atoms['H'] = X_atoms['N'] + _normalize(
            _normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
        +  _normalize(X_atoms['N'] - X_atoms['CA'], -1)
    , -1)

    def _distance(X_a, X_b):
        return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

    def _inv_distance(X_a, X_b):
        return 1. / (_distance(X_a, X_b) + eps)

    U = (0.084 * 332) * (
            _inv_distance(X_atoms['O'], X_atoms['N'])
        + _inv_distance(X_atoms['C'], X_atoms['H'])
        - _inv_distance(X_atoms['O'], X_atoms['H'])
        - _inv_distance(X_atoms['C'], X_atoms['N'])
    )

    HB = (U < -0.5).type(torch.float32)
    neighbor_HB = mask_neighbors * gather_edges_d(HB.unsqueeze(-1),  E_idx)
    return neighbor_HB

def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges_d(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6) #[B, L, L]
        RBF_A_B = _rbf(D_A_B, num_rbf)
    return RBF_A_B

def _orientations_coarse_gl(X, E_idx, eps=1e-6):
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0 = n_0[:,::3,:]
    b_1 = b_1[:,::3,:]
    X = X[:,::3,:]

    O = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0,0,0,1), 'constant', 0) # [16, 464, 9]

    O_neighbors = gather_nodes_d(O, E_idx) # [16, 464, 30, 9]
    X_neighbors = gather_nodes_d(X, E_idx) # [16, 464, 30, 3]

    O = O.view(list(O.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

    dX = X_neighbors - X.unsqueeze(-2) # [16, 464, 30, 3]
    dU = torch.matmul(O, dX.unsqueeze(-1)).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
    R = torch.matmul(O.transpose(-1,-2), O_neighbors)
    feat = torch.cat((_normalize(dU, dim=-1), _quaternions(R)), dim=-1) # 相对方向向量+旋转四元数
    return feat


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    #N CA C O
    V = X.clone()
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) #B 3L 3
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0 = n_0[:,::3,:]
    b_1 = b_1[:,::3,:]
    X = X[:,::3,:]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [16, 464, 9]

    Q_neighbors = gather_nodes_d(Q, E_idx) # [16, 464, 30, 9]
    X_neighbors = gather_nodes_d(V[:,:,1,:], E_idx) # [16, 464, 30, 3]
    N_neighbors = gather_nodes_d(V[:,:,0,:], E_idx)
    C_neighbors = gather_nodes_d(V[:,:,2,:], E_idx)
    O_neighbors = gather_nodes_d(V[:,:,3,:], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
    Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

    dX = torch.stack([X_neighbors,N_neighbors,C_neighbors,O_neighbors], dim=3) - X[:,:,None,None,:] # [16, 464, 30, 3]
    dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K,-1)
    R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
    q = _quaternions(R)
    # edge_feat = torch.cat((dU, q), dim=-1) # 相对方向向量+旋转四元数
    
    dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B,N,-1)
    return V_direct, E_direct, q

def gather_edges_d(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes_d(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) # [4, 317, 30]-->[4, 9510]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [4, 9510, dim]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat) # [4, 9510, dim]
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1]) # [4, 317, 30, 128]


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz, 
        - Rxx + Ryy - Rzz, 
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)



def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av
def loss_smoothed_4(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 4).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        

        
        self.all_global = EncLayer_global(num_hidden)
        



    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, res_idx = None):
        """ Parallel computation of full transformer layer """
        E_num = E_idx.shape[-1]

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / E_num#self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            
        h_V = self.all_global(h_V,mask_V,res_idx)
        
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        
        
        #c_V = h_V.mean(1)
        #h_V = h_V * (self.V_MLP_g(c_V).unsqueeze(1))
        
        
        
        
        return h_V, h_E

    


class EncLayer_global(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=None):
        super(EncLayer_global, self).__init__()
        self.num_hidden = num_hidden
        #self.num_in = num_in
        self.dropout1 = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(num_hidden)
        #self.norm2 = nn.LayerNorm(num_hidden)
        self.pos = SinusoidalPositionalEmbedding(num_hidden,0)


        #self.act = torch.nn.GELU()
        #self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        

        self.att = TransformerBlock(num_hidden,num_hidden*4)



    def forward(self, h_V,mask,res_idx):
        """ Parallel computation of full transformer layer """
        
        pos_hid = self.pos(res_idx.squeeze(-1))

        h_V_agg = self.att(h_V+pos_hid,mask)
        
        #h_V_agg = self.act(h_V_agg)
        
        h_V = self.norm1(h_V + self.dropout1(h_V_agg))

                
        return h_V


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        
        E_num = h_E.shape[-2]
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / E_num

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


def mask_tensor(A, mask):
    
    mask = mask.unsqueeze(-1).expand_as(A)
    
    A = A * mask
    
    return A

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


def replace_masked_elements(A, B, mask):
    
    assert A.shape == B.shape and A.shape[:2] == mask.shape, "SSSSSSS"
    
    
    mask = mask.unsqueeze(-1).expand_as(A)
    
    
    result = torch.where(mask == 0, B, A)
    
    return result

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        
        edge_in = edge_in+ 18
        
        node_in = 22
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        
        self.node_embedding = nn.Linear(node_in, edge_features, bias=False)
        self.norm_node = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx
    
    def _dist_seq(self, res_idx):
 
        
        D_adjust = res_idx[:,:,None] - res_idx[:,None,:]
        D_adjust = abs(D_adjust)
        sampled_top_k = self.top_k
        _, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, res_idx.shape[1]), dim=-1, largest=False)
        return  E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        #N CA C O
        N,B = X.shape[:2]
        
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)
        E_num = E_idx.shape[-1]
        
        E_idx_res = self._dist_seq(residue_idx)
        E_idx = replace_masked_elements(E_idx,E_idx_res,mask)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        
        
        V_angles = _dihedrals(X, 0)  # B N 12
        #V_angles = node_mask_select(V_angles)
        
        mask_V_angles = mask.unsqueeze(-1)
        V_angles = V_angles * mask_V_angles
    
        

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        
        V_direct = V_direct * mask_V_angles #B N 9
        
        mask_E = torch.gather(mask.unsqueeze(1).repeat(1,B,1),-1,E_idx) #B N E
        
        mask_expend = mask.unsqueeze(-1).repeat(1,1,E_num)
        
        mask_expend_E = torch.cat([mask_E.unsqueeze(-1),mask_expend.unsqueeze(-1)],-1) #B N E 2
        
        E_direct = E_direct * (mask_E.unsqueeze(-1))#B N E 12
        E_angles = E_angles * (mask_E.unsqueeze(-1))#B N E 4
        
        
        
        

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        
        E = torch.cat((E, E_direct), -1)
        
        E = torch.cat((E, E_angles), -1)
        
        E = torch.cat((E, mask_expend_E), -1)
        
        
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        V = torch.cat([V_angles,V_direct],-1)
        
        V = torch.cat([V,mask.unsqueeze(-1)],-1)
        
        V = self.node_embedding(V)
        V = self.norm_node(V)
        
        return E, E_idx,V

class ProDualNet(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProDualNet, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_v = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        

        
        #self.node_comb_layers = nn.ModuleList([node_comb(hidden_dim),node_comb(hidden_dim)])
        
        self.W_out= nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_out1 = nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_out2 = nn.Linear(hidden_dim, num_letters, bias=True)
        
        
        self.W_low1 = nn.Linear(1280, 512, bias=True)
        self.W_low2 = nn.Linear(512, 128, bias=True)
        self.norm_esm = nn.LayerNorm(128)
        self.W_esm = nn.Linear(256, 128, bias=True)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S,S_embed,mask_train, mask, chain_M, residue_idx, chain_encoding_all,  is_eval = False ):
        chain_M_1 = chain_M.detach().clone()
        device=X.device
        
        B ,L= X.shape[:2]
        E, E_idx,h_V = self.features(X, mask_train, residue_idx, chain_encoding_all)
        
        h_E = self.W_e(E)
        h_V = self.W_v(h_V)
        
        esm_bed = self.norm_esm(self.W_low2(self.W_low1(S_embed)))
        
        h_V = self.W_esm(torch.cat([esm_bed,h_V],-1))
        

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        #print(mask)
        ns = 1
        for layer in self.encoder_layers:
            # print(mask.shape)
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend, residue_idx)
            
            h_V = h_V.reshape(B // 2, 2, L, self.hidden_dim)

            h_V_res = torch.cat([h_V[:, 1][:, None], h_V[:, 0][:, None]], 1)
                
            h_V = 0.8 * h_V + 0.2* h_V_res * (chain_M_1.reshape(B // 2, 2, L).unsqueeze(-1)) + \
                      0.2 * h_V * (torch.ones_like(chain_M_1.reshape(B // 2, 2, L).unsqueeze(-1)) - chain_M_1.reshape(B // 2, 2,L).unsqueeze(-1))
               
            h_V = h_V.reshape(B, L, self.hidden_dim)
            ns += 1

        
        h_S = self.W_s(S)
        
      
        
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        
        if is_eval:

            chain_M = chain_M*mask #update chain_M to include missing regions
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            chain_M_d = - (chain_M - torch.ones_like(chain_M))
            chain_M_d = chain_M_d.unsqueeze(-2).repeat(1,chain_M.shape[-1],1)
            
            order_mask_backward = torch.ones_like(order_mask_backward) * chain_M_d
            
            
        
        else:
            
            chain_M = chain_M*mask #update chain_M to include missing regions
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        ns = 0
        for layer in self.decoder_layers:
            
            
            
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

            

        logits = self.W_out(h_V)
        
        
        B,L = logits.shape[0],logits.shape[1]
        
        logits = logits.reshape(B//2,2,L,21).mean(1)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        
        
        return log_probs
    def sample(self, X, randn, S_true, S_embed, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=0.2,chain_M_pos=None,mask_train = None):
        device = X.device
        
        chain_M_1 = chain_mask.detach().clone()
        B ,L= X.shape[:2]
        E, E_idx,h_V = self.features(X, mask_train, residue_idx, chain_encoding_all)
        
        h_E = self.W_e(E)
        h_V = self.W_v(h_V)
        
        esm_bed = self.norm_esm(self.W_low2(self.W_low1(S_embed)))
        
        h_V = self.W_esm(torch.cat([esm_bed,h_V],-1))

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        #print(mask)
        ns = 1
        for layer in self.encoder_layers:
            # print(mask.shape)
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend, residue_idx)
            
            
            h_V = h_V.reshape(B // 2, 2, L, self.hidden_dim)

            h_V_res = torch.cat([h_V[:, 1][:, None], h_V[:, 0][:, None]], 1)
                # print(chain_M.reshape(B//2,2,L).shape,h_V_res.shape)
            h_V = 0.8 * h_V + 0.2* h_V_res * (chain_M_1.reshape(B // 2, 2, L).unsqueeze(-1)) + \
                      0.2 * h_V * (torch.ones_like(chain_M_1.reshape(B // 2, 2, L).unsqueeze(-1)) - chain_M_1.reshape(B // 2, 2,L).unsqueeze(-1))
               
            h_V = h_V.reshape(B, L, self.hidden_dim)
            ns += 1


        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        #for ns in range(len(chain_mask)-1):
        chain_mask = chain_mask[0]*chain_mask[1]#*chain_mask[2]
        #print(chain_mask.sum())
        chain_mask = chain_mask[None,:].repeat(2,1)
        
        
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
         
        decoding_order = decoding_order[0][None,:].repeat(2,1)
        
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        



        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder#b n k c
        #print(chain_M_pos*chain_M)
        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]
            #print(t)
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
            #bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:,None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                
                if (chain_M_pos*chain_mask)[0,t[0]] == 1:
                    h_V_t = h_V_t.mean(0)
                    #print(h_V_t.shape)
                    h_V_t = h_V_t[None,:].repeat(2,1)
                    logits = self.W_out(h_V_t) / temperature
                
                else:
                    logits = self.W_out(h_V_t) / temperature
                
                
                probs = F.softmax(logits, dim=-1)

                
                
                
                S_t = torch.multinomial(probs, 1)
                
                if (chain_M_pos*chain_mask)[0,t[0]] == 1:
                    probs = probs.mean(0)[None,:].repeat(2,1)
                    S_t = torch.multinomial(probs, 1)[0][None,:].repeat(2,1) 
                     
                
                
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        
        
        return output_dict["S"],output_dict["probs"]