# -*- coding: utf-8 -*-
"""CGCNN 模型定义（供 train_cgcnn.py / train_cgcnn_mp.py / predict_cgcnn.py 共用）。

⚠️ 本模块在**导入时**就需要 torch；所有调用方都必须把 ``import torch`` 与
``from cgcnn_model import build_model`` 放在 ``main()`` 里（而不是模块顶层），
这样 ``--help`` 等纯参数解析路径在没有 torch 的环境（例如 CI）仍然可用。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cgcnn_data import ATOM_FEAT_DIM, EDGE_FEAT_DIM


class ConvLayer(nn.Module):
    """CGCNN 门控卷积（扁平边索引 + index_add 聚合，不 padding）。"""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_feat, src, dst):
        total = torch.cat([x[src], x[dst], edge_feat], dim=1)
        full = self.bn1(self.fc_full(total))
        gate, core = torch.chunk(full, 2, dim=1)
        msg = self.sigmoid(gate) * self.softplus(self.bn2(core))
        aggr = torch.zeros_like(x).index_add_(0, dst, msg)
        return x + aggr


class CrystalGraphConvNet(nn.Module):
    """节点嵌入 → n_conv 层门控卷积 → 晶体内平均池化 → MLP → 标量回归。"""

    def __init__(self, atom_fea_len: int = 64, n_conv: int = 3,
                 h_fea_len: int = 128, n_h: int = 1):
        super().__init__()
        self.embedding = nn.Linear(ATOM_FEAT_DIM, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, EDGE_FEAT_DIM) for _ in range(n_conv)])
        self.fc_in = nn.Linear(atom_fea_len, h_fea_len)
        self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h)])
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.softplus = nn.Softplus()

    def forward(self, atom_fea, edge_fea, edge_src, edge_dst, batch_idx):
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, edge_fea, edge_src, edge_dst)
        n_crystals = int(batch_idx.max().item()) + 1
        pooled = torch.zeros(n_crystals, x.shape[1], device=x.device).index_add_(0, batch_idx, x)
        counts = torch.zeros(n_crystals, device=x.device).index_add_(
            0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        pooled = pooled / counts.unsqueeze(1)
        h = self.softplus(self.fc_in(pooled))
        for fc in self.fcs:
            h = self.softplus(fc(h))
        return self.fc_out(h).squeeze(1)


def build_model(atom_fea_len: int = 64, n_conv: int = 3, h_fea_len: int = 128,
                n_h: int = 1) -> CrystalGraphConvNet:
    return CrystalGraphConvNet(atom_fea_len, n_conv, h_fea_len, n_h)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

def batch_tensors(graphs, idx, device, target=None):
    """把一批图拼成 (atom_fea, edge_fea, edge_src, edge_dst, batch_idx) 张量。

    Args:
        graphs: cgcnn_data.build_graph 产出的图 dict 列表
        idx: 本批次在图列表中的下标
        device: torch.device
        target: 可选，与 idx 对齐的目标值数组 -> 追加返回 y
    """
    import numpy as np

    node, efea, src, dst, bvec = [], [], [], [], []
    a_off = 0
    for b, i in enumerate(idx):
        g = graphs[i]
        n = int(g["n_atoms"])
        node.append(torch.as_tensor(g["node_feat"]))
        efea.append(torch.as_tensor(g["edge_feat"]))
        src.append(torch.as_tensor(g["edge_src"] + a_off))
        dst.append(torch.as_tensor(g["edge_dst"] + a_off))
        bvec.append(torch.full((n,), b, dtype=torch.long))
        a_off += n
    out = (torch.cat(node), torch.cat(efea), torch.cat(src), torch.cat(dst), torch.cat(bvec))
    out = tuple(x.to(device, non_blocking=True) for x in out)
    if target is None:
        return out
    return out, torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
