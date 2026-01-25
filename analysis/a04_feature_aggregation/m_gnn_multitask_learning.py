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
    
class MultiTaskGraphDataset(Dataset):
    """loading graph data for multi-task learning
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
        # ---------------- Load label dict ----------------
        if "train" in self.mode or "valid" in self.mode:
            subject_info, label_dict = self.info_list[idx]
        else:
            subject_info = self.info_list[idx]
            label_dict = None

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

                    x, edge_index = self.load_npz_cached(npz_path)

                    if self.preproc:
                        proc = self.preproc.get(f"{key}_{child_name}")
                        if proc:
                            x = proc(x)
                            x = torch.as_tensor(x, dtype=torch.float32)

                    if key == "radiomics":
                        x, edge_index = self.sample_subgraph(x, edge_index)

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
                edge_cat = torch.cat(entry["edges"], dim=1)

                node_type = f"{key}_{child_name}"
                hetero[node_type].x = x_cat
                hetero[(node_type, "to", node_type)].edge_index = edge_cat

        # ---------------- Handle labels (multi-task) ----------------
        if label_dict is not None:
            # ---- Survival ----
            surv_tasks = list(label_dict["survival"].keys())
            y_surv = []
            mask_surv = []
            for task in surv_tasks:
                duration = label_dict["survival"][task]["duration"]
                event = label_dict["survival"][task]["event"]
                if duration is None or event is None:
                    y_surv.append([0.0, 0.0])
                    mask_surv.append(0)
                else:
                    y_surv.append([float(duration), float(event)])
                    mask_surv.append(1)
            hetero['survival'].y = torch.tensor(y_surv, dtype=torch.float32).unsqueeze(0)
            hetero['survival'].mask = torch.tensor(mask_surv, dtype=torch.float32).unsqueeze(0)

            # ---- Classification ----
            cls_tasks = list(label_dict["classification"].keys())
            y_cls = []
            mask_cls = []
            for task in cls_tasks:
                cls_val = label_dict["classification"][task]["class"]
                if cls_val is None:
                    y_cls.append(-1)  # placeholder
                    mask_cls.append(0)
                else:
                    y_cls.append(int(cls_val))
                    mask_cls.append(1)
            hetero['classification'].y = torch.tensor(y_cls, dtype=torch.long).unsqueeze(0)
            hetero['classification'].mask = torch.tensor(mask_cls, dtype=torch.float32).unsqueeze(0)

            # ---- Regression ----
            reg_tasks = list(label_dict["regression"].keys())
            y_reg = []
            mask_reg = []
            for task in reg_tasks:
                sub_dict = label_dict["regression"][task]
                for k, v in sub_dict.items():
                    if v is None:
                        y_reg.append(0.0)
                        mask_reg.append(0)
                    else:
                        y_reg.append(float(v))
                        mask_reg.append(1)
            hetero['regression'].y = torch.tensor(y_reg, dtype=torch.float32).unsqueeze(0)
            hetero['regression'].mask = torch.tensor(mask_reg, dtype=torch.float32).unsqueeze(0)

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
                LayerNorm(out_dims),
                Dropout(self.dropout),
                Linear(out_dims, out_dims)
            )
        
        input_emb_dim = dim_features
        # self.head = create_block(input_emb_dim, dim_features)
        for out_emb_dim in [dim_features, dim_hidden]:
            self.pools.append(TopKPooling(input_emb_dim, ratio=pool_ratio)) 
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
            input_emb_dim = out_emb_dim
        # self.tail = create_block(input_emb_dim, dim_hidden)

    def forward(self, feature, edge_index, batch):
        # feature = self.head(feature)
        # N = maybe_num_nodes(batch)
        # feature_mean = scatter(feature, index=batch, dim=0, dim_size=N, reduce='mean')
        # feature_mean = feature_mean.index_select(0, batch)
        # feature = feature - feature_mean

        perm_list = []
        for l in range(2):
            feature, edge_index, _, batch, perm, score = self.pools[l](feature, edge_index, batch=batch)
            perm_list.append(perm)
            if self.conv_name in ["MLP"]:
                feature = self.convs[l](feature)
            else:
                feature = self.convs[l](feature, edge_index)
        
        # feature = self.tail(feature)
        
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

class MultiTaskGraphArch(nn.Module):
    """define Graph architecture for multi-task learning
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
            task_to_idx,
            dropout=0.0,
            conv="GINConv",
            aggregation="SPARRA",
            num_groups=1,
            mu0=0, 
            lambda0=1, 
            alpha0=2, 
            beta0=1e-2,
            **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.aggregation = aggregation
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.task_to_idx = task_to_idx

        self.dim_dict = {f"{mod}_{child}": val
            for mod, children in dim_features.items()
            for child, val in children.items()
        }
        self.pool_dict = {f"{mod}_{child}": val
            for mod, children in pool_ratio.items()
            for child, val in children.items()
        }

        self.conv_name = conv

        if aggregation == "MEAN":
            self.Aggregation = MeanAggregation()
        elif aggregation == "ABMIL":
            self.gate_nn_dict = nn.ModuleDict(
                {
                    k: Attn_Net_Gated(L=v, D=256, dropout=0.25, n_classes=v)
                    for k, v in self.dim_dict.items()
                }
            )
            self.Aggregation = SumAggregation()
        elif aggregation == "SPARRA":
            self.gate_nn_dict = nn.ModuleDict(
                {
                    k: Attn_Net_Gated(L=v, D=256, dropout=0.25, n_classes=v)
                    for k, v in self.dim_dict.items()
                }
            )
            self.downsample_nn_dict = nn.ModuleDict(
                {
                    k: DownsampleArch(
                        dim_features=v,
                        dim_hidden=2*v,
                        dropout=0.0,
                        pool_ratio=self.pool_dict[k],
                        conv=self.conv_name
                    )
                    for k, v in self.dim_dict.items()
                }
            )
            self.hetero_gate_dict = nn.ModuleDict(
                {
                    k: Linear(v, 1, bias=True)
                    for k, v in self.dim_dict.items()
                }
            )
            self.MeanAggregation = MeanAggregation()
            self.SumAggregation = SumAggregation()
            self.omega = nn.Parameter(
                torch.tensor([0.8, 0.2]).unsqueeze(0).repeat(len(self.dim_dict), 1)
            )
        else:
            raise NotImplementedError

        self.predictor_dict = nn.ModuleDict(
            {
                k: Linear(num_groups * v, dim_target, bias=True)
                for k, v in self.dim_dict.items()
            }
        )
    
        feature_dim = sum(list(self.dim_dict.values()))
        self.predictor = Linear(num_groups * feature_dim, dim_target, bias=True)

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print("Loading failed:", e)
            pass

    def sampling(self, data):
        loc, logvar = torch.chunk(data, chunks=2, dim=-1)
        logvar = torch.clamp(logvar, min=-20, max=0)
        gauss = torch.distributions.Normal(loc, torch.exp(0.5*logvar))
        return gauss.sample(), loc, logvar

    def precision(self, loc, logvar):
        alpha = self.alpha0 + 0.5
        beta = self.beta0 + 0.5*(self.lambda0*(loc - self.mu0)**2 + self.lambda0*torch.exp(logvar))
        # beta = beta.clone().detach()
        return alpha, beta

    def group_sparsemax(self, x, index):
        out = torch.zeros_like(x)
        for i in index.unique():
            mask = (index == i)
            out[mask] = self.sparsemax(x[mask])
        return out

    def forward(self, data):
        if self.aggregation == "MEAN":
            gate_dict = {}
            
            # feature aggregation
            feature_dict = {
                k: self.Aggregation(data.x_dict[k], index=data.batch_dict[k])
                for k in self.dim_dict.keys()
            }
            VIparas = None
            perm_dict = None
        elif self.aggregation == "ABMIL":
            gate_dict = {}
            for k in self.dim_dict.keys():
                gate = self.gate_nn_dict[k](data.x_dict[k])
                gate = softmax(gate, index=data.batch_dict[k], dim=-2)
                gate_dict.update({k: gate})
            
            # feature aggregation
            feature_dict = {
                k: self.Aggregation(data.x_dict[k] * gate_dict[k], index=data.batch_dict[k])
                for k in self.dim_dict.keys()
            }
            VIparas = None
            perm_dict = None
        elif self.aggregation == "SPARRA":
            # encoder
            feature_dict, batch_dict = {}, {}
            perm_dict, score_list = {}, []
            for k in self.dim_dict.keys():
                feature = self.gate_nn_dict[k](data.x_dict[k])
                feature, edge_index, batch, perm_list, score = self.downsample_nn_dict[k](
                    feature = feature, 
                    edge_index = data.edge_index_dict[k, "to", k], 
                    batch = data.batch_dict[k]
                )
                feature_dict.update({k: feature})
                batch_dict.update({k: batch})
                perm_dict.update({k: perm_list})

            # sparsity-informed aggregation
            sample_dict, loc_dict, logvar_dict = {}, {}, {}
            gate_dict, precision_dict = {}, {}
            for k in self.dim_dict.keys():
                # sparse_score = self.gate_nn_dict[k](feature_dict[k])
                sample, loc, logvar = self.sampling(feature_dict[k])
                sample_dict.update({k: sample})
                loc_dict.update({k: loc})
                logvar_dict.update({k: logvar})
                alpha, beta = self.precision(loc, logvar)
                precision = alpha / beta # smaller, more uncertain

                precision_prior_mode = (self.alpha0 - 1) / self.beta0
                # w = min(0.2, 0.01 + epoch * 0.02)
                # w = max(0.4, 3 - epoch * 0.5)
                # threshold = precision.mean().detach().clamp(
                #     max=0.2 * precision_prior_mode
                # )
                # gate = torch.relu(precision - threshold)
                gate = torch.relu(precision - 0.2 * precision_prior_mode)

                sparsity = (gate == 0).float().mean(axis=0)
                s_min = sparsity.min().item()
                s_mean = sparsity.mean().item()
                s_max = sparsity.max().item()
                print(f'{k} Sparsity:', (s_min, s_mean, s_max))
                # gate = F.softplus(precision - 0.2 * precision_prior_mode)
                # gate = precision * torch.sigmoid(0.05 * (precision - precision_prior_mode))
                batch = batch_dict[k]
                N = maybe_num_nodes(batch)
                gate_sum = scatter(gate, index=batch, dim=0, dim_size=N, reduce='sum')
                gate_sum = gate_sum.index_select(0, batch)
                gate_dict.update({k: gate / (gate_sum + 1e-6)})
                precision_dict.update({k: [alpha / beta, self.lambda0, self.mu0]})

                # get inverse precison
                # batch = batch_dict[k]
                # N = maybe_num_nodes(batch)
                # inv_precision_sum = scatter(inv_precision, index=batch, dim=0, dim_size=N, reduce='sum')
                # inv_precision_sum = inv_precision_sum.max(dim=1, keepdim=True)[0]
                # inv_precision_sum = inv_precision_sum.mean(dim=1, keepdim=True)
                # inv_precision_sum = inv_precision_sum.index_select(0, batch)
                # gate = inv_precision / inv_precision_sum
                # gate_dict.update({k: gate})
                # precision_dict.update({k: [alpha / beta, self.lambda0, self.mu0]})

            VIparas = [loc_dict, logvar_dict, precision_dict]

            # get references of downsampled graphs
            ds_enc_dict = {}
            for k in self.dim_dict.keys():
                enc = data.x_dict[k]
                perm_list = perm_dict[k]
                for p in perm_list: 
                    enc = enc[p]
                ds_enc_dict.update({k: enc})
            VIparas.append(ds_enc_dict)
            VIparas.append(sample_dict)
            
            # get homogeneous features
            homo_feature_dict = {
                k: self.MeanAggregation(data.x_dict[k], index=data.batch_dict[k])
                for k in self.dim_dict.keys()
            }

            # get heterogeneous features
            # if self.training:
            #     feature_dict = sample_dict
            # else:
            #     feature_dict = loc_dict
            hetero_feature_dict = {
                k: self.SumAggregation(ds_enc_dict[k] * gate_dict[k], index=batch_dict[k])
                for k in self.dim_dict.keys()
            }
            
            # get fused features
            # hetero_gate = {
            #     k: torch.sigmoid(self.hetero_gate_dict[k](hetero_feature_dict[k] - homo_feature_dict[k]))
            #     for k in self.dim_dict.keys()
            # }
            # feature_dict = {
            #     k: (1 - hetero_gate[k]) * homo_feature_dict[k] + hetero_gate[k] * hetero_feature_dict[k]
            #     for k in self.dim_dict.keys()
            # }
            # print('Hetero gate:', [(k, v.min().item(), v.max().item()) for k, v in hetero_gate.items()])
            # omega = torch.softmax(self.omega, dim=1)
            # feature_dict = {
            #     k: omega[i, 0] * homo_feature_dict[k] + omega[i, 1] * hetero_feature_dict[k]
            #     for i, k in enumerate(self.dim_dict.keys())
            # }
            # print("Omega:", omega[:, 0].min().item(), omega[:, 1].max().item())
            feature_dict = {
                k: 0.0 * homo_feature_dict[k] + 1.0 * hetero_feature_dict[k]
                for k in self.dim_dict.keys()
            }
        
        # normalize feature due to different scales for different graphs
        feature_dict = {
            k: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 16
            for k, v in feature_dict.items()
        }

        # graph-specific predictions for training the aggregation modules
        pred_dict = {
            k: self.predictor_dict[k](feature_dict[k])
            for k in self.dim_dict.keys()
        }

        # only used for training a predictor, so detached
        feature = torch.concat(
            [feature_dict[k] for k in self.dim_dict.keys()],
            dim=-1
        ).detach()
        prediction = self.predictor(feature)
        
        return prediction, pred_dict, VIparas, feature, perm_dict, gate_dict
    
    @staticmethod
    def train_batch(model, batch_data, on_gpu, loss, optimizer, kl=None, l1_penalty=0.0):
        device = "cuda" if on_gpu else "cpu"
        batch_data = batch_data.to(device)

        model.train()
        optimizer.zero_grad()
        predictions, outputs, VIparas, _, _, _ = model(batch_data)
        labels = batch_data.y_dict
        masks = batch_data.mask_dict
        loss = loss(predictions, outputs, labels, masks, VIparas)

        # add regularization
        loss += l1_penalty * model.predictor.weight.abs().sum()
        if model.predictor.bias is not None:
            loss += 10 * l1_penalty * model.predictor.bias.abs().sum()
        for predictor in model.predictor_dict.values():
            loss += l1_penalty * predictor.weight.abs().sum()
            if predictor.bias is not None:
                loss += 10 * l1_penalty * predictor.bias.abs().sum()

        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)
        labels = {k: v.cpu().numpy() for k, v in labels.items()}
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
        labels, masks = None, None
        try:
            labels = batch_data.y_dict
            labels = {k: v.cpu().numpy() for k, v in labels.items()}
            masks = batch_data.mask_dict
            masks = {k: v.cpu().numpy() for k, v in masks.items()}
        except:
            pass
        if labels is not None and masks is not None:
            return [outputs, labels, masks]
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

class MultiTaskBayesGraphArch(nn.Module):
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
    def __init__(self, tau_ae=1e-2):
        super(VILoss, self).__init__()
        self.tau_ae = tau_ae

    def forward(self, loc, logvar, pre, enc, dec):
        precision, lambda0, mu0 = pre
        precision = precision.clone().detach()
        pre_min = precision.min().item()
        pre_mean = precision.mean().item()
        pre_max = precision.max().item()

        loss_kl = 0.5*torch.mean(lambda0*precision*((loc - mu0)**2 + torch.exp(logvar)) - logvar)
        loss_ae = F.mse_loss(enc - torch.mean(enc, dim=0, keepdim=True), dec)
        print(loss_ae.item(), loss_kl.item(), (pre_min, pre_mean, pre_max))

        return self.tau_ae * loss_ae + loss_kl

# class VILoss(nn.Module):
#     """ variational inference loss
#     """
#     def __init__(self, tau_ae=10):
#         super(VILoss, self).__init__()
#         self.tau_ae = tau_ae

#     def forward(self, loc_dict, logvar_dict, pre_dict, enc_dict, dec_dict):
#         loss_ae = 0.0
#         loss_kl = 0.0
#         pre_min = []
#         pre_max = []
#         sparsity = []
#         for k in loc_dict.keys():
#             loc, logvar = loc_dict[k], logvar_dict[k]
#             precision, lambda0, mu0 = pre_dict[k]
#             precision = precision.clone().detach()
#             pre_min.append(precision.min().item())
#             pre_max.append(precision.max().item())
#             sparsity.append((precision > 100).float().mean().item())

#             loss_kl += 0.5*torch.mean(lambda0*precision*((loc - mu0)**2 + torch.exp(logvar)) - logvar)
#             enc, dec = enc_dict[k], dec_dict[k]
#             loss_ae += F.mse_loss(enc - torch.mean(enc, dim=0, keepdim=True), dec)
#         loss_ae = loss_ae / len(dec_dict)
#         loss_kl = loss_kl / len(dec_dict)
#         print(loss_ae.item(), loss_kl.item(), min(pre_min), max(pre_max), min(sparsity), max(sparsity))
#         return self.tau_ae * loss_ae + loss_kl
    
class CoxSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hazards, time, event):
        """
        hazards: (B, 1) or (B,)
        time:    (B,)
        event:   (B,)  {1=event, 0=censored}
        """

        device = hazards.device
        hazards = hazards.view(-1)
        time = time.view(-1)
        event = event.view(-1)

        # Skip if no events in batch
        if event.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Sort by descending time
        order = torch.argsort(time, descending=True)
        hazards = hazards[order]
        event = event[order]

        # log cumulative sum exp (numerically stable)
        log_cumsum_exp = torch.logcumsumexp(hazards, dim=0)

        # Cox loss
        loss = -(hazards - log_cumsum_exp) * event
        return loss.sum() / event.sum()

    
class MultiTaskLoss(nn.Module):
    def __init__(self, task_to_idx, tau_vi=0.1, tau_surv=1.0, tau_cls=1.0, tau_reg=1.0):
        super(MultiTaskLoss, self).__init__()
        self.task_to_idx = task_to_idx
        self.tau_vi = tau_vi
        self.tau_surv = tau_surv
        self.tau_cls = tau_cls
        self.tau_reg = tau_reg

        self.loss_surv = CoxSurvLoss()
        self.loss_vi = VILoss()
        self.loss_cls = nn.CrossEntropyLoss(reduction='none')
        self.loss_reg = nn.MSELoss(reduction='none')

    def forward(self, logits, y_dict, mask_dict, graph_idx, VIparas=None):
        """
        logits: model output [B, N_total_subtasks]
        y_dict: HeteroData labels, with keys 'survival', 'classification', 'regression'
        VIparas: optional parameters for VI loss
        """
        survival_loss = 0.0
        classification_loss = 0.0
        regression_loss = 0.0
        num_tasks = 0
        precision_dict = {}

        # ---------------- Survival ----------------
        if 'survival' in y_dict:
            y_surv = y_dict['survival']          # [B, num_surv_tasks, 2]
            mask_surv = mask_dict['survival']   # same shape
            for i, task in enumerate(self.task_to_idx['survival'].keys()):

                # Mask valid entries
                m = mask_surv[:, i] > 0
                if m.sum() == 0:
                    continue
                
                dur = y_surv[m, i, 0]
                ev = y_surv[m, i, 1]
                idx = self.task_to_idx['survival'][task]

                # Compute loss
                l_i = self.loss_surv(logits[m, idx], dur, ev)
                survival_loss += self.tau_surv * l_i
                num_tasks += 1
            # print('survival loss:', survival_loss.item())

        # ---------------- Classification ----------------
        if 'classification' in y_dict:
            y_cls = y_dict['classification']      # [B, num_cls_tasks]
            mask_cls = mask_dict['classification']  # same shape
            for i, task in enumerate(self.task_to_idx['classification'].keys()):
                idx_info = self.task_to_idx['classification'][task]

                # Handle binary vs multi-class
                if isinstance(idx_info, tuple):
                    idx_s, idx_e = idx_info
                    logits_task = logits[:, idx_s:idx_e]  # [B, num_classes]
                else:
                    idx = idx_info
                    logits_task = logits[:, idx].unsqueeze(1)  # [B, 1]

                # Mask valid entries
                m = mask_cls[:, i] > 0
                if m.sum() == 0:
                    continue

                targets = y_cls[m, i].long()  # ensure long type for CrossEntropyLoss
                logits_valid = logits_task[m]

                # Compute loss
                l_i = self.loss_cls(logits_valid, targets)
                classification_loss += self.tau_cls * l_i.sum() / m.sum()
                num_tasks += 1
            # print('classification loss:', classification_loss.item())

        # ---------------- Regression ----------------
        if 'regression' in y_dict:
            y_reg = y_dict['regression']        # [B, num_reg_tasks, n_subcols]
            mask_reg = mask_dict['regression']  # same shape
            for i, task in enumerate(self.task_to_idx['regression'].keys()):
                idx = self.task_to_idx['regression'][task]

                # Extract targets and mask
                targets = y_reg[:, i]           # [B, n_subcols] or [B] if single column
                m = mask_reg[:, i] > 0          # [B,] boolean

                if m.sum() == 0:
                    continue

                logits_valid = logits[m, idx]   # [num_valid, ...]
                targets_valid = targets[m]      # [num_valid, ...]

                # Compute loss
                l_i = self.loss_reg(logits_valid, targets_valid)
                regression_loss += self.tau_reg * l_i.sum() / m.sum()
                num_tasks += 1
            # print('regression loss:', regression_loss.item())

        # multi-task loss
        total_loss = (survival_loss + classification_loss + regression_loss) / num_tasks

        # ---------------- VI Loss (optional) ----------------
        if VIparas is not None:
            loss_vi = self.loss_vi(*VIparas)
            total_loss += self.tau_vi * loss_vi

        return total_loss, precision_dict

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, task_to_idx, graph_to_idx, tau_vi=0.1):
        super(WeightedMultiTaskLoss, self).__init__()
        self.task_to_idx = task_to_idx
        self.tau_vi = tau_vi

        self.subtask_idx = (
            list(task_to_idx['survival'].keys()) +
            list(task_to_idx['classification'].keys()) +
            list(task_to_idx['regression'].keys())
        )
        self.graph_to_idx = list(graph_to_idx) + ['integration']
        self.log_sigma = nn.Parameter(
            torch.zeros([len(self.graph_to_idx), len(self.subtask_idx)])
        )

        self.loss_surv = CoxSurvLoss()
        self.loss_vi = VILoss()
        self.loss_cls = nn.CrossEntropyLoss(reduction='none')
        self.loss_reg = nn.MSELoss(reduction='none')
    
    def get_log_sigma(self, graph_name, task_name):
        i = self.graph_to_idx.index(graph_name)
        j = self.subtask_idx.index(task_name)
        return self.log_sigma[i,j]

    def forward(self, logits, y_dict, mask_dict, graph_idx, VIparas=None):
        """
        logits: model output [B, N_total_subtasks]
        y_dict: HeteroData labels, with keys 'survival', 'classification', 'regression'
        VIparas: optional parameters for VI loss
        """
        survival_loss = 0.0
        classification_loss = 0.0
        regression_loss = 0.0
        num_tasks = 0
        precision_dict = {}

        # ---------------- Survival ----------------
        if 'survival' in y_dict:
            y_surv = y_dict['survival']          # [B, num_surv_tasks, 2]
            mask_surv = mask_dict['survival']   # same shape
            for i, task in enumerate(self.task_to_idx['survival'].keys()):

                # Mask valid entries
                m = mask_surv[:, i] > 0
                if m.sum() == 0:
                    continue
                
                dur = y_surv[m, i, 0]
                ev = y_surv[m, i, 1]
                idx = self.task_to_idx['survival'][task]

                # Compute loss
                l_i = self.loss_surv(logits[m, idx], dur, ev)
                log_sigma = self.get_log_sigma(graph_idx, task)
                precision = torch.exp(-log_sigma)
                precision_dict.update({task: precision.item()})
                survival_loss += precision * l_i + log_sigma
                num_tasks += 1

        # ---------------- Classification ----------------
        if 'classification' in y_dict:
            y_cls = y_dict['classification']      # [B, num_cls_tasks]
            mask_cls = mask_dict['classification']  # same shape
            for i, task in enumerate(self.task_to_idx['classification'].keys()):
                idx_info = self.task_to_idx['classification'][task]

                # Handle binary vs multi-class
                if isinstance(idx_info, tuple):
                    idx_s, idx_e = idx_info
                    logits_task = logits[:, idx_s:idx_e]  # [B, num_classes]
                else:
                    idx = idx_info
                    logits_task = logits[:, idx].unsqueeze(1)  # [B, 1]

                # Mask valid entries
                m = mask_cls[:, i] > 0
                if m.sum() == 0:
                    continue

                targets = y_cls[m, i].long()  # ensure long type for CrossEntropyLoss
                logits_valid = logits_task[m]

                # Compute loss
                l_i = self.loss_cls(logits_valid, targets)
                log_sigma = self.get_log_sigma(graph_idx, task)
                precision = torch.exp(-log_sigma)
                precision_dict.update({task: precision.item()})
                classification_loss += precision * l_i.sum() / m.sum() + log_sigma
                num_tasks += 1

        # ---------------- Regression ----------------
        if 'regression' in y_dict:
            y_reg = y_dict['regression']        # [B, num_reg_tasks, n_subcols]
            mask_reg = mask_dict['regression']  # same shape
            for i, task in enumerate(self.task_to_idx['regression'].keys()):
                idx = self.task_to_idx['regression'][task]

                # Extract targets and mask
                targets = y_reg[:, i]           # [B, n_subcols] or [B] if single column
                m = mask_reg[:, i] > 0          # [B,] boolean

                if m.sum() == 0:
                    continue

                logits_valid = logits[m, idx]   # [num_valid, ...]
                targets_valid = targets[m]      # [num_valid, ...]

                # Compute loss
                l_i = self.loss_reg(logits_valid, targets_valid)
                log_sigma = self.get_log_sigma(graph_idx, task)
                precision = torch.exp(-log_sigma)
                precision_dict.update({task: precision.item()})
                regression_loss += precision * l_i.sum() / m.sum() + log_sigma
                num_tasks += 1

        # multi-task loss
        total_loss = (survival_loss + classification_loss + regression_loss) / num_tasks

        # ---------------- VI Loss (optional) ----------------
        if VIparas is not None:
            loss_vi = self.loss_vi(*VIparas)
            total_loss += self.tau_vi * loss_vi

        return total_loss, precision_dict

class MultiPredEnsembleLoss(nn.Module):
    def __init__(self, task_to_idx, graph_to_idx, task_weighted=True, lambda_fuse=1.0):
        super(MultiPredEnsembleLoss, self).__init__()
        self.steps = 0
        self.lambda_fuse = lambda_fuse
        if not task_weighted:
            self.multi_task_loss = MultiTaskLoss(task_to_idx)
        else:
            self.multi_task_loss = WeightedMultiTaskLoss(task_to_idx, graph_to_idx)

    def forward(self, predictions, pred_dict, y_dict, mask_dict, VIparas=None):
        loss = 0.0
        self.steps += 1
        for i, k in enumerate(pred_dict.keys()):
            if VIparas is not None:
                VI = [p[k] for p in VIparas]
            else:
                VI = None
            mt_loss, mt_precision_dict = self.multi_task_loss(pred_dict[k], y_dict, mask_dict, k, VI)
            loss += mt_loss
            if self.steps % 20 == 0:
                print(f'\nGraph {k} multi-task weights:', mt_precision_dict)

        # add loss for fused predictions
        mt_loss, _ = self.multi_task_loss(predictions, y_dict, mask_dict, 'integration')
        loss += self.lambda_fuse * mt_loss

        return loss
