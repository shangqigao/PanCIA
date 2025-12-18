import json
import joblib
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbnn as bnn

from torch.nn import BatchNorm1d, Linear, ReLU, Dropout, LayerNorm, GroupNorm, GELU
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.utils import subgraph, softmax, scatter
import torch_geometric.utils as pyg_utils
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import EdgeConv, GINConv, GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, TopKPooling
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, AttentionalAggregation

from functools import lru_cache
from sparsemax import Sparsemax
    
class SurvivalGraphDataset(Dataset):
    """loading graph data for survival analysis
    """
    def __init__(
        self,
        info_list,
        mode="train",
        preproc=None,
        data_types=["radiomics", "pathomics"],
        sampling_rate=1.0,
        max_num_nodes=1e3,
        sampling_k=2,      # <-- new parameter: BFS k-hop neighborhood
        cache_size=2048,              # bounded LRU cache
    ):
        super().__init__()
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc
        self.data_types = data_types
        self.sampling_rate = sampling_rate
        self.max_num_nodes = max_num_nodes
        self.sampling_k = sampling_k

        # Create a cached loader with bounded memory
        self.load_npz_cached = lru_cache(maxsize=cache_size)(self._load_npz_internal)

    # ----------------------------------------------------------------------
    # Cached NPZ loader (fast)
    # ----------------------------------------------------------------------
    def _load_npz_internal(self, npz_path):
        data = np.load(npz_path, mmap_mode="r")  # memory-mapped read
        x = torch.from_numpy(data["x"]).float()
        edge_index = torch.from_numpy(data["edge_index"]).long()
        return x, edge_index

    # ----------------------------------------------------------------------
    # Optional k-hop sampling
    # ----------------------------------------------------------------------
    def sample_subgraph(self, x, edge_index):
        assert x.size(0) > edge_index.max(), "Edge index contains invalid node indices"
        num_nodes = x.size(0)

        if self.sampling_rate >= 1 or num_nodes <= self.max_num_nodes:
            return x, edge_index

        num_sampled = int(num_nodes * self.sampling_rate)
        seed_count = max(1, num_sampled // 10)

        seeds = torch.randperm(num_nodes)[:seed_count]

        subset, new_edge_index, _, _ = pyg_utils.k_hop_subgraph(
            seeds,
            self.sampling_k,
            edge_index,
            num_nodes=num_nodes,
            relabel_nodes=True
        )

        return x[subset], new_edge_index

    # ----------------------------------------------------------------------
    # Main loading
    # ----------------------------------------------------------------------
    def get(self, idx):
        # ---------------- Label handling ----------------
        if "train" in self.mode or "valid" in self.mode:
            subject_info, label = self.info_list[idx]
            label = torch.tensor(label).unsqueeze(0)
        else:
            subject_info = self.info_list[idx]

        hetero = HeteroData()
        subject_paths = subject_info[1]

        # ---------------- Iterate through data types ----------------
        for key in self.data_types:
            parents = subject_paths.get(key)
            if parents is None:
                continue

            child_nodes = {}

            for parent in parents:
                for child_name, npz_path in parent.items():
                    # ---- Load from NPZ (fast, cached) ----
                    x, edge_index = self.load_npz_cached(npz_path)

                    # ---- Preprocessing (applied once per epoch load) ----
                    if self.preproc:
                        proc = self.preproc.get(f"{key}_{child_name}")
                        if proc:
                            x = proc(x)
                            x = torch.as_tensor(x, dtype=torch.float32)

                    # ---- Optional sampling ----
                    x, edge_index = self.sample_subgraph(x, edge_index)

                    # ---- Group by child_name ----
                    entry = child_nodes.setdefault(
                        child_name, {"xs": [], "edges": [], "offset": 0}
                    )

                    offset = entry["offset"]
                    entry["xs"].append(x)
                    entry["edges"].append(edge_index + offset)
                    entry["offset"] += x.size(0)

            # ---------------- Merge children into HeteroData ----------------
            for child_name, entry in child_nodes.items():
                x_cat = torch.cat(entry["xs"], dim=0)
                # mean = x_cat.mean(dim=0, keepdim=True)
                # std = x_cat.std(dim=0, keepdim=True)
                # std[std == 0] = 1.0
                # x_cat_norm = (x_cat - mean) / std
                edge_cat = torch.cat(entry["edges"], dim=1)

                node_type = f"{key}_{child_name}"
                hetero[node_type].x = x_cat
                hetero[(node_type, "to", node_type)].edge_index = edge_cat

                if "train" in self.mode or "valid" in self.mode:
                    hetero[node_type].y = label

        return hetero

    def len(self):
        return len(self.info_list)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Bayes_Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1, Bayes_std=0.1):
        super(Bayes_Attn_Net_Gated, self).__init__()
        self.attention_a = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [
            # bnn.BayesLinear(0, Bayes_std, L, D),
            nn.Linear(L, D),
            nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = bnn.BayesLinear(0, Bayes_std, D, n_classes)
        # self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class Score_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Score_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D)]

        self.attention_b = [nn.Linear(L, D)]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
    
class ImportanceScoreArch(nn.Module):
    """define importance score architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.num_layers = len(self.embedding_dims)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                # LayerNorm(out_dims),
                Dropout(self.dropout),
                Linear(out_dims, out_dims)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        # self.final_norm = LayerNorm(self.embedding_dims[-1])
        self.tail = Linear(self.embedding_dims[-1], dim_target)  

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embedding_dims[1:]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
                self.linears.append(Linear(out_emb_dim, out_emb_dim))
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(create_block(out_emb_dim, out_emb_dim))
            else:
                subnet = create_block(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
                self.linears.append(nn.Sequential())
                
            input_emb_dim = out_emb_dim

    def forward(self, feature, edge_index):
        feature = self.head(feature)
        for layer in range(1, self.num_layers):
            if self.conv_name in ["MLP"]:
                feature = self.convs[layer - 1](feature)
            else:
                feature = self.convs[layer - 1](feature, edge_index)
            feature = self.linears[layer - 1](feature)
        # feature = self.final_norm(feature)
        output = self.tail(feature)
        return output
    
class DownsampleArch(nn.Module):
    """define downsample architecture
    """
    def __init__(
            self, 
            dim_features, 
            dim_hidden,
            dropout=0.0,
            pool_ratio=0.1,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                # LayerNorm(out_dims),
                Dropout(self.dropout),
                Linear(out_dims, out_dims)
            )
        
        input_emb_dim = dim_features
        for out_emb_dim in [dim_hidden, dim_hidden]:
            conv_class, alpha = conv_dict[self.conv_name]
            if self.conv_name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * input_emb_dim, out_emb_dim)
                subnet = conv_class(block, **kwargs)
                self.convs.append(subnet)
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
            else:
                subnet = create_block(alpha * input_emb_dim, out_emb_dim)
                self.convs.append(subnet)
            self.pools.append(TopKPooling(out_emb_dim, ratio=pool_ratio))    
            input_emb_dim = out_emb_dim

    def forward(self, feature, edge_index, batch):
        perm_list = []
        for l in range(2):
            feature, edge_index, _, batch, perm, score = self.pools[l](feature, edge_index, batch=batch)
            perm_list.append(perm)
            if self.conv_name in ["MLP"]:
                feature = self.convs[l](feature)
            else:
                feature = self.convs[l](feature, edge_index)
        return feature, edge_index, batch, perm_list, score
    
class OmicsIntegrationArch(nn.Module):
    """define importance score architecture
    """
    def __init__(
            self, 
            dim_teachers, 
            dim_student,
            dim_target,
            dropout=0.0,
            conv="GINConv",
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [Linear, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims, normalize=False):
            if normalize:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    LayerNorm(out_dims),
                    ReLU(),
                    Dropout(self.dropout)
                )
            else:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    ReLU(),
                    Dropout(self.dropout)
                )
        
        def conv_block(in_dims, out_dims, name="GCNConv", normalize=False):
            conv_class, alpha = conv_dict[name]
            if name in ["GINConv", "EdgeConv"]:
                block = create_block(alpha * in_dims, out_dims, normalize)
                subnet = conv_class(block, **kwargs)
            elif self.conv_name in ["GCNConv", "GATConv"]:
                subnet = conv_class(alpha * in_dims, out_dims)
            else:
                subnet = create_block(alpha * in_dims, out_dims, normalize)
            return subnet
        
        self.align_teachers = nn.ModuleList()
        for ch_t in dim_teachers:
            self.align_teachers.append(
                conv_block(ch_t, 2*dim_target, self.conv_name, normalize=True)
            )

        self.align_student = conv_block(dim_student, 2*dim_target, self.conv_name, normalize=True)

        self.extractor = conv_block(2*dim_target, dim_target, self.conv_name, normalize=True)

        self.recon_teachers = nn.ModuleList()
        for ch_t in dim_teachers:
            self.recon_teachers.append(
                conv_block(dim_target, ch_t, self.conv_name, normalize=False)
            )

    def forward(self, ft, et, fs, es):
        assert len(ft) == len(et)
        aligned_t = [self.align_teachers[i](ft[i], et[i]) for i in range(len(ft))]
        aligned_s = self.align_student(fs, es)
        ht = [self.extractor(f, e) for f, e in zip(aligned_t, et)]
        hs = self.extractor(aligned_s, es)
        ft_ = [self.recon_teachers[i](ht[i], et[i]) for i in range(len(ht))]
        ht = [torch.concat(ht, axis=0)]
        return (hs, ht), (ft_, ft)

class SurvivalGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    Args:
        aggregation: attention-based multiple instance learning (ABMIL) 
        or sparsity informed spatial regression and aggregation (SPARRA)
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers,
            pool_ratio,
            dropout=0.0,
            conv="GINConv",
            aggregation="SPARRA",
            num_groups=1,
            mu0=0, 
            lambda0=1, 
            alpha0=2, 
            beta0=1e-4,
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dims = layers
        self.aggregation = aggregation
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.dim_dict = {f"{mod}_{child}": val
            for mod, children in dim_features.items()
            for child, val in children.items()
        }
        self.pool_dict = {f"{mod}_{child}": val
            for mod, children in pool_ratio.items()
            for child, val in children.items()
        }

        self.num_layers = len(self.embedding_dims)
        self.enc_convs = nn.ModuleList()
        self.enc_linears = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_linears = nn.ModuleList()
        self.conv_name = conv

        self.sparsemax = Sparsemax(dim=0)

        def create_block(in_dims, out_dims, normalize=False):
            if normalize:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    ReLU(),
                    LayerNorm(out_dims),
                    Dropout(self.dropout),
                    Linear(out_dims, out_dims)
                )
            else:
                return nn.Sequential(
                    Linear(in_dims, out_dims),
                    ReLU(),
                    # Dropout(self.dropout),
                    # Linear(out_dims, out_dims)
                )
       
        out_emb_dim = self.embedding_dims[0]
        self.enc_branches = nn.ModuleDict(
            {
                k: create_block(v, out_emb_dim) 
                for k, v in self.dim_dict.items()
            }
        )
        self.dec_branches = nn.ModuleDict(
            {
                k: Linear(out_emb_dim, v) 
                for k, v in self.dim_dict.items()
            }
        )
        input_emb_dim = out_emb_dim

        if aggregation == "ABMIL":
            self.gate_nn_dict = nn.ModuleDict(
                {
                    k: Attn_Net_Gated(L=input_emb_dim, D=256, dropout=0.25, n_classes=1)
                    for k in self.dim_dict.keys()
                }
            )
            self.Aggregation = SumAggregation()
            # self.final_norm = LayerNorm(num_groups*len(self.dim_dict)*out_emb_dim)
            self.final_norm = GroupNorm(num_groups, num_groups*len(self.dim_dict)*out_emb_dim)
        elif aggregation == "SPARRA":
            hid_emb_dim = self.embedding_dims[1]
            self.mean_nn_branch = create_block(input_emb_dim, hid_emb_dim)
            self.downsample_nn_dict = nn.ModuleDict(
                {
                    k: DownsampleArch(
                        dim_features=input_emb_dim,
                        dim_hidden=2 * hid_emb_dim,
                        dropout=0.0,
                        pool_ratio=self.pool_dict[k],
                        conv=self.conv_name
                    )
                    for k in self.dim_dict.keys()
                }
            )
            out_emb_dim = hid_emb_dim
            input_emb_dim = hid_emb_dim
            # self.gate_nn = Attn_Net_Gated(
            #     L=input_emb_dim, 
            #     D=256, 
            #     dropout=0.25, 
            #     n_classes=1
            # )
            hid_emb_dim = self.embedding_dims[0]
            self.inverse_score_nn = ImportanceScoreArch(
                dim_features=input_emb_dim,
                dim_target=hid_emb_dim,
                layers=self.embedding_dims[1:],
                dropout=0.0,
                conv=self.conv_name
            )
            self.MeanAggregation = MeanAggregation()
            self.SumAggregation = SumAggregation()
            self.final_norm = GroupNorm(num_groups, num_groups*len(self.dim_dict)*out_emb_dim)
        else:
            raise NotImplementedError

        self.classifier = Linear(num_groups*len(self.dim_dict)*out_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def sampling(self, data):
        loc, logvar = torch.chunk(data, chunks=2, dim=-1)
        logvar = torch.clamp(logvar, min=-20, max=20)
        gauss = torch.distributions.Normal(loc, torch.exp(0.5*logvar))
        return gauss.sample(), loc, logvar

    def precision(self, loc, logvar):
        alpha = self.alpha0 + 0.5
        beta = self.beta0 + 0.5*(self.lambda0*(loc - self.mu0)**2 + self.lambda0*torch.exp(logvar))
        beta = beta.detach()
        return alpha, beta

    def group_sparsemax(self, x, index):
        out = torch.zeros_like(x)
        for i in index.unique():
            mask = (index == i)
            out[mask] = self.sparsemax(x[mask])
        return out

    def forward(self, data):
        feature_dict = {k : self.enc_branches[k](data.x_dict[k]) for k in self.dim_dict.keys()}
        batch_dict = {k: data.batch_dict[k] for k in self.dim_dict.keys()}

        if self.aggregation == "ABMIL":
            gate_dict = {}
            for k in self.dim_dict.keys():
                gate = self.gate_nn_dict[k](feature_dict[k])
                gate = softmax(gate, index=batch_dict[k], dim=-2)
                gate_dict.update({k: gate})
            
            # feature aggregation
            feature_list = [
                self.Aggregation(feature_dict[k] * gate_dict[k], index=batch_dict[k])
                for k in self.dim_dict.keys()
            ]
            VIparas = None
            KAparas = None
            perm_dict = None
            feature = torch.concat(feature_list, dim=-1)
        elif self.aggregation == "SPARRA":
            # get homogeneous features
            homo_feature_list = [
                self.MeanAggregation(self.mean_nn_branch(feature_dict[k]), index=batch_dict[k])
                for k in self.dim_dict.keys()
            ]
            # encoder
            feature_list, edge_index_list, batch_list = [], [], []
            perm_dict, score_list = {}, []
            for k in self.dim_dict.keys():
                feature, edge_index, batch, perm_list, score = self.downsample_nn_dict[k](
                    feature = feature_dict[k], 
                    edge_index = data.edge_index_dict[k, "to", k], 
                    batch = batch_dict[k]
                )
                feature_list.append(feature)
                edge_index_list.append(edge_index)
                batch_list.append(batch)
                perm_dict.update({k: perm_list})
                score_list.append(score)
            
            # graph concatenation
            feature = torch.concat(feature_list, dim=0)
            edge_index = [e.clone() for e in edge_index_list]
            ins = [0]
            index_shift = 0
            for i in range(1, len(self.dim_dict)): 
                index_shift += len(feature_list[i - 1])
                edge_index[i] += index_shift
                ins.append(index_shift)
            ins.append(len(feature))
            edge_index = torch.concat(edge_index, dim=1)
            batch = torch.concat(batch_list, dim=0)
            batch_dict = {k: b for k, b in zip(self.dim_dict.keys(), batch_list)}
            score = torch.concat(score_list, dim=0)
            KAparas = None

            # reparameterization
            sample, loc, logvar = self.sampling(feature)
            VIparas = [loc, logvar]

            # caculate normalized inverse precision
            alpha, beta = self.precision(loc, logvar)
            inv_precision = beta / (alpha - 1)  # mean of inverse precision/gamma
            gate = inv_precision

            # caculate raw moments
            if self.training:
                feature = sample
            else:
                feature = loc
            # feature = sample  # the first raw moment
            # feature = torch.concat([feature, loc**2 + inv_precision], dim=-1)  # the second raw moment
            # feature = torch.concat([feature, loc**3 + 3*loc*inv_precision], dim=-1)  # the third raw moment
            # feature = torch.concat([feature, loc**4 + 6*loc**2*inv_precision + 3*inv_precision*beta/(alpha-2)], dim=-1)  # the fourth raw moment
            # gate = self.group_sparsemax(inv_precision, index=batch)
            # N = maybe_num_nodes(batch)
            # inv_precision_sum = scatter(inv_precision, index=batch, dim=-2, dim_size=N, reduce='sum')
            # inv_precision_sum = inv_precision_sum.index_select(-2, batch)
            # gate = inv_precision / inv_precision_sum
            # if self.training:
            #     gate = self.group_sparsemax(sample, index=batch)
            # else:
            #     gate = self.group_sparsemax(loc, index=batch)
            precision_list = [alpha / beta, self.lambda0, self.mu0]
            VIparas.append(precision_list)

            
            feature_dict = {k: feature[ins[i]:ins[i+1]] for i, k in enumerate(self.dim_dict.keys())}
            gate_dict = {k: gate[ins[i]:ins[i+1]] for i, k in enumerate(self.dim_dict.keys())}

            # decoder
            decode = self.inverse_score_nn(sample, edge_index)

            dec_list = [
                self.dec_branches[k](decode[ins[i]:ins[i+1], ...]) 
                for i, k in enumerate(self.dim_dict.keys())
            ]

            # get references of downsampled graphs
            ds_enc_list = []
            for k in self.dim_dict.keys():
                enc = data.x_dict[k]
                perm_list = perm_dict[k]
                for p in perm_list: 
                    enc = enc[p]
                ds_enc_list.append(enc)
            VIparas.append(ds_enc_list)
            VIparas.append(dec_list)
            
            # get heterogeneous features
            hetero_feature_list = [
                self.MeanAggregation(feature_dict[k], index=batch_dict[k])
                for k in self.dim_dict.keys()
            ]
            feature = torch.concat(homo_feature_list + hetero_feature_list, dim=-1)
        
        # outcome prediction
        # feature = self.final_norm(feature)
        output = self.classifier(feature)

        return output, VIparas, KAparas, feature, perm_dict, gate_dict
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl=None, l1_penalty=0.0):
        device = "cuda" if on_gpu else "cpu"
        batch_data = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        outputs, VIparas, KAparas, _, _, _ = model(batch_data)
        outputs = outputs.squeeze()
        labels = next(iter(batch_data.y_dict.values())).squeeze()
        loss = loss(outputs, labels[:, 0], labels[:, 1], VIparas, KAparas)
        loss += l1_penalty * model.classifier.weight.abs().sum()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        labels = labels.cpu().numpy()
        return [loss, outputs, labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        batch_data = batch_data.to(device)

        model.eval()
        with torch.inference_mode():
            outputs, _, _, features, perm_dict, gate_dict = model(batch_data)
        outputs = outputs.cpu().numpy()
        features = features.cpu().numpy()
        labels = None
        try:
            labels = next(iter(batch_data.y_dict.values())).cpu().numpy()
        except:
            pass
        if labels is not None:
            return [outputs, labels]
        else:
            if perm_dict is not None:
                atte_dict = {}
                for k in model.dim_dict.keys():
                    enc = batch_data.x_dict[k]
                    msk_list = []
                    msk = torch.zeros(enc.shape[0], gate_dict[k].shape[-1]).to(enc.device)
                    msk_list.append(msk)
                    perm_list = perm_dict[k]
                    for p in perm_list: 
                        enc = enc[p]
                        msk = torch.zeros(enc.shape[0], gate_dict[k].shape[-1]).to(enc.device)
                        msk_list.append(msk)
                    for n, msk in enumerate(msk_list[::-1]):
                        if n == 0:
                            atte = gate_dict[k]
                        else:
                            perm = perm_list[-n]
                            msk[perm] = atte
                            atte = msk
                    atte_dict.update({k: atte.cpu().numpy()})
            else:
                atte_dict = {k: v.cpu().numpy() for k, v in gate_dict.items()}

            if outputs.shape[0] == 1:
                return [outputs, features, atte_dict]
            else:
                return [outputs, features]

class SurvivalBayesGraphArch(nn.Module):
    """define Graph architecture for survival analysis
    """
    def __init__(
            self, 
            dim_features, 
            dim_target,
            layers=None,
            dropout=0.0,
            conv="GINConv",
            Bayes_std=0.1,
            **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embedding_dims = layers
        self.num_layers = len(self.embedding_dims)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.conv_name = conv

        conv_dict = {
            "MLP": [None, 1],
            "GCNConv": [GCNConv, 1],
            "GATConv": [GATv2Conv, 1],
            "GINConv": [GINConv, 1], 
            "EdgeConv": [EdgeConv, 2]
        }
        if self.conv_name not in conv_dict:
            raise ValueError(f"Not support conv={conv}.")
        
        def create_block(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims),
                ReLU(),
                Dropout(self.dropout)
            )
        
        input_emb_dim = dim_features
        out_emb_dim = self.embedding_dims[0]
        self.head = create_block(input_emb_dim, out_emb_dim)
        input_emb_dim = out_emb_dim
        out_emb_dim = self.embedding_dims[-1]

        # input_emb_dim = out_emb_dim
        # for out_emb_dim in self.embedding_dims[1:]:
        #     conv_class, alpha = conv_dict[self.conv_name]
        #     if self.conv_name in ["GINConv", "EdgeConv"]:
        #         block = create_block(alpha * input_emb_dim, out_emb_dim)
        #         subnet = conv_class(block, **kwargs)
        #         self.convs.append(subnet)
        #         self.linears.append(bnn.BayesLinear(0, Bayes_std, input_emb_dim, out_emb_dim))
        #     elif self.conv_name in ["GCNConv", "GATConv"]:
        #         subnet = conv_class(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(create_block(out_emb_dim, out_emb_dim))
        #     else:
        #         subnet = create_block(alpha * input_emb_dim, out_emb_dim)
        #         self.convs.append(subnet)
        #         self.linears.append(nn.Sequential())
                
        #     input_emb_dim = out_emb_dim
            
        self.gate_nn = Bayes_Attn_Net_Gated(
            L=input_emb_dim,
            D=out_emb_dim,
            dropout=0.25,
            n_classes=1,
            Bayes_std=Bayes_std
        )
        self.global_attention = AttentionalAggregation(
            gate_nn=self.gate_nn,
            nn=None
        )
        self.classifier = Linear(input_emb_dim, dim_target)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, data):
        feature, edge_index, batch = data.x, data.edge_index, data.batch

        feature = self.head(feature)
        # for layer in range(1, self.num_layers):
        #     feature = F.dropout(feature, p=self.dropout, training=self.training)
        #     if self.conv_name in ["MLP"]:
        #         feature = self.convs[layer - 1](feature)
        #     else:
        #         feature = self.convs[layer - 1](feature, edge_index)
        #     feature = self.linears[layer - 1](feature)

        feature = self.global_attention(feature, index=batch)
        output = self.classifier(feature)
        return output
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)

        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.train()
        optimizer.zero_grad()
        wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.squeeze()
        wsi_labels = wsi_graphs.y.squeeze()
        loss = loss(wsi_outputs, wsi_labels[:, 0], wsi_labels[:, 1])
        kl_loss, kl_weight = kl["loss"](model)[0], kl["weight"]
        loss = loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_outputs, wsi_labels]
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        wsi_graphs = batch_data.to(device)
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        model.eval()
        with torch.inference_mode():
            wsi_outputs = model(wsi_graphs)
        wsi_outputs = wsi_outputs.cpu().numpy()
        if wsi_graphs.y is not None:
            wsi_labels = wsi_graphs.y.cpu().numpy()

            return [wsi_outputs, wsi_labels]
        return [wsi_outputs]

class ScalarMovingAverage:
    """Class to calculate running average."""

    def __init__(self, alpha=0.95):
        """Initialize ScalarMovingAverage."""
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self, step_output):
        """ScalarMovingAverage instances behave and can be called like a function."""
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # Calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # Init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value

class VILoss(nn.Module):
    """ variational inference loss
    """
    def __init__(self, tau_ae=10):
        super(VILoss, self).__init__()
        self.tau_ae = tau_ae

    def forward(self, loc, logvar, pre_list, enc_list, dec_list):
        precision, lambda0, mu0 = pre_list
        precision = precision.clone().detach()
        loss_kl = 0.5*torch.mean(lambda0*precision*((loc - mu0)**2 + torch.exp(logvar)) - logvar)
        loss_ae = 0.0
        for enc, dec in zip(enc_list, dec_list):
            # subset = torch.randperm(len(enc))[:len(dec)]
            # enc = enc[subset]
            loss_ae += F.mse_loss(enc - torch.mean(enc, dim=0, keepdim=True), dec)
        loss_ae = loss_ae / len(dec_list)
        print(loss_ae.item(), loss_kl.item(), precision.min().item(), precision.max().item())
        return self.tau_ae * loss_ae + loss_kl
    
def calc_mmd(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return mmd_rbf2(f1, f2, sigmas=sigmas)


def mmd_rbf2(x, y, sigmas=None):
    N, _ = x.shape
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = L = P = 0.0
    XX2 = rx.t() + rx - 2*xx
    YY2 = ry.t() + ry - 2*yy
    XY2 = rx.t() + ry - 2*zz

    if sigmas is None:
        sigma2 = torch.mean((XX2.detach()+YY2.detach()+2*XY2.detach()) / 4)
        sigmas2 = [sigma2/4, sigma2/2, sigma2, sigma2*2, sigma2*4]
        alphas = [1.0 / (2 * sigma2) for sigma2 in sigmas2]
    else:
        alphas = [1.0 / (2 * sigma**2) for sigma in sigmas]

    for alpha in alphas:
        K += torch.exp(- alpha * (XX2.clamp(min=1e-12)))
        L += torch.exp(- alpha * (YY2.clamp(min=1e-12)))
        P += torch.exp(- alpha * (XY2.clamp(min=1e-12)))

    beta = (1./(N*(N)))
    gamma = (2./(N*N))

    return F.relu(beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P))

class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht, ft_, ft):
        mmd_loss = 0.0
        mse_loss = 0.0
        # random sampling if hs is large-scale
        if len(hs) > 1e5:
            num_sampled = int(len(hs) * 0.01)
            subset = torch.randperm(len(hs))[:num_sampled]
            hs = hs[subset]
            ht = [ht_i[subset] for ht_i in ht]
        for ht_i in ht:
            mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])
        
        return mmd_loss + self.beta*mse_loss
    
class CoxSurvLoss(nn.Module):
    def __init__(self, tau_vi=0.1, tau_ka=1e-2):
        super(CoxSurvLoss, self).__init__()
        self.tau_vi = tau_vi
        self.tau_ka = tau_ka
        self.loss_vi = VILoss()
        self.loss_ka = CFLoss()

    def forward(self, hazards, time, c, VIparas, KAparas):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        # c = torch.tensor(c).to(hazards.device)
        R_mat = torch.tensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * c)
        if VIparas is not None:
            loss_vi = self.loss_vi(*VIparas)
            loss_cox = loss_cox + self.tau_vi * loss_vi
        if KAparas is not None:
            loss_ka = self.loss_ka(*KAparas)
            loss_cox = loss_cox + self.tau_ka * loss_ka
        return loss_cox
    
