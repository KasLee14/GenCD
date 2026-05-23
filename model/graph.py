# model/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import csv
import ast


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(
        torch.from_numpy(indices).long(),
        torch.from_numpy(values).float(),
        torch.Size(graph.shape),
    )
    return graph.coalesce()


class HeteroGraphPrior(nn.Module):
    def __init__(self, config):
        super(HeteroGraphPrior, self).__init__()
        self.config = config
        self.device = config.device
        self.embedding_size = config.embedding_size
        self.num_layers = config.num_layers

        self.num_users, self.num_exercises, self.num_knowledge = self.get_data_size(config.dataset_adr)

        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.exercises_feature = nn.Parameter(torch.FloatTensor(self.num_exercises, self.embedding_size))
        nn.init.xavier_normal_(self.exercises_feature)
        self.knowledge_feature = nn.Parameter(torch.FloatTensor(self.num_knowledge, self.embedding_size))
        nn.init.xavier_normal_(self.knowledge_feature)

        # 构建并缓存隐式超图传播所需的所有稀疏组件
        self.build_hypergraph_components(config.dataset_adr)

        self.sigmoid = nn.Sigmoid()

        self.view1 = None
        self.view2 = None

    def get_data_size(self, adr):
        with open(adr + "datasize.txt", 'r') as f:
            return [int(s) + 1 for s in f.readline().split('\t')]

    def build_hypergraph_components(self, adr):
        """
        构建基于 U-E-K 的三元超图的独立稀疏组件，避免稠密展开
        """
        row, col, data = [], [], []

        with open(adr + "train.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for s in reader:
                u, e = int(s[0]), int(s[1])
                row.append(u)
                col.append(e)
                data.append(1.0)

        for e in range(self.num_exercises):
            row.append(self.num_users + e)
            col.append(e)
            data.append(1.0)

        with open(adr + "item.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                e = int(r[0])
                k_list = ast.literal_eval(r[1])
                for k in k_list:
                    row.append(self.num_users + self.num_exercises + int(k))
                    col.append(e)
                    data.append(1.0)

        total_nodes = self.num_users + self.num_exercises + self.num_knowledge

        # 关联矩阵 H
        H = sp.coo_matrix((data, (row, col)), shape=(total_nodes, self.num_exercises))
        Dv = np.array(H.sum(axis=1)).flatten()
        De = np.array(H.sum(axis=0)).flatten()

        # 将对角阵直接降维为列向量，用于后续的高效逐元素广播乘法 (Element-wise multiplication)
        Dv_inv_sqrt = np.power(Dv + 1e-8, -0.5).reshape(-1, 1)
        De_inv = np.power(De + 1e-8, -1.0).reshape(-1, 1)

        # 转化为 tensor 并挂载到设备
        self.H = to_tensor(H).to(self.device)
        self.H_T = to_tensor(H.T).to(self.device)
        self.Dv_inv_sqrt = torch.FloatTensor(Dv_inv_sqrt).to(self.device)
        self.De_inv = torch.FloatTensor(De_inv).to(self.device)

    def implicit_hgcn_forward(self, X):
        """
        隐式超图传播：时间复杂度从 O(V^2) 极速降维至 O(|E|)
        执行顺序：X -> 乘 Dv_inv_sqrt -> 左乘 H^T -> 乘 De_inv -> 左乘 H -> 乘 Dv_inv_sqrt
        """
        # 1. 节点特征度归一化 (V, dim)
        X = X * self.Dv_inv_sqrt
        # 2. 从节点聚合到超边 (E, dim)
        E_features = torch.spmm(self.H_T, X)
        # 3. 超边特征度归一化 (E, dim)
        E_features = E_features * self.De_inv
        # 4. 从超边广播回节点 (V, dim)
        V_features = torch.spmm(self.H, E_features)
        # 5. 再次节点特征度归一化 (V, dim)
        out = V_features * self.Dv_inv_sqrt
        return out

    def propagate(self, is_train=False):
        ego_emb = torch.cat([self.users_feature, self.exercises_feature, self.knowledge_feature], dim=0)
        all_emb = ego_emb
        layer_embs = [ego_emb]

        if is_train:
            # 【修复 2】：在 Layer 0 也必须注入噪声，否则跨层求平均时会被相同的初始特征稀释
            norm_ego = F.normalize(ego_emb, p=2, dim=1)
            noise1_0 = torch.rand_like(norm_ego) * self.config.eps - self.config.eps / 2
            noise2_0 = torch.rand_like(norm_ego) * self.config.eps - self.config.eps / 2
            all_emb1 = norm_ego + noise1_0
            all_emb2 = norm_ego + noise2_0
            layer_embs1 = [all_emb1]
            layer_embs2 = [all_emb2]

        for i in range(self.num_layers):
            all_emb = self.implicit_hgcn_forward(all_emb)
            layer_embs.append(all_emb)

            if is_train:
                emb1 = self.implicit_hgcn_forward(all_emb1)
                emb2 = self.implicit_hgcn_forward(all_emb2)

                # 归一化特征，锚定相同的几何尺度
                norm_emb1 = F.normalize(emb1, p=2, dim=1)
                norm_emb2 = F.normalize(emb2, p=2, dim=1)

                noise1 = torch.rand_like(norm_emb1) * self.config.eps - self.config.eps / 2
                noise2 = torch.rand_like(norm_emb2) * self.config.eps - self.config.eps / 2

                # 【修复 1】：必须严格将噪声加在【归一化后】的特征上，确保扰动比例显著！
                all_emb1 = norm_emb1 + noise1
                all_emb2 = norm_emb2 + noise2

                layer_embs1.append(all_emb1)
                layer_embs2.append(all_emb2)

        final_emb = torch.mean(torch.stack(layer_embs, dim=1), dim=1)

        if is_train:
            final_emb1 = torch.mean(torch.stack(layer_embs1, dim=1), dim=1)
            final_emb2 = torch.mean(torch.stack(layer_embs2, dim=1), dim=1)

            u1, e1, _ = torch.split(final_emb1, [self.num_users, self.num_exercises, self.num_knowledge], dim=0)
            u2, e2, _ = torch.split(final_emb2, [self.num_users, self.num_exercises, self.num_knowledge], dim=0)
            self.view1 = (u1, e1)
            self.view2 = (u2, e2)

        users_feat, exercises_feat, knowledge_feat = torch.split(
            final_emb, [self.num_users, self.num_exercises, self.num_knowledge], dim=0)

        return users_feat, exercises_feat, knowledge_feat
