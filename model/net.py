# model/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import HeteroGraphPrior
from model.gce import GCE


def info_nce_loss(z1, z2, temp=0.1):
    """
    璁＄畻 InfoNCE Loss (鎵规鍐呰礋鏍锋湰)
    """
    # 銆愪慨澶?3銆戯細濡傛灉褰撳墠 Batch 鍘婚噸鍚庡彧鍓╀笅 1 涓嫭绔嬭妭鐐癸紝鏃犳硶鏋勬垚璐熸牱鏈帹寮€锛岀洿鎺ヨ繑鍥?0
    if z1.size(0) < 2:
        return torch.tensor(0.0, device=z1.device)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(logits.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.device = config.device
        self.embedding_size = config.embedding_size

        self.graph_prior = HeteroGraphPrior(config)
        self.gce = GCE(config)

        self.mask_token = nn.Parameter(torch.zeros(1, self.embedding_size))
        nn.init.xavier_normal_(self.mask_token.unsqueeze(0))

        self.irt_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Dropout(config.dropout_ratio),
            nn.Linear(self.embedding_size, 1)
        )

        for name, param in self.named_parameters():
            if 'weight' in name and 'feature' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)

    def forward(self, user_id, seq_item_id, seq_score, seq_time, target_item_id, target_knowledge_emb, target_time,
                item_mask=None):
        # 1. 寮傛瀯瓒呭浘鐗瑰緛棰勭儹 (骞剁敓鎴愬姣旇鍥?
        stu_feature, e_feature, k_feature = self.graph_prior.propagate(is_train=self.training)

        stu_emb = stu_feature[user_id]
        seq_item_emb = e_feature[seq_item_id]
        e_disc_emb = e_feature[target_item_id]

        # ==========================================
        # 璁＄畻鍥惧姣斿涔犳崯澶?(Graph Contrastive Loss)
        # ==========================================
        graph_cl_loss = torch.tensor(0.0, device=self.device)
        if self.training:
            u1, e1 = self.graph_prior.view1
            u2, e2 = self.graph_prior.view2

            # 銆愭牳蹇冧慨澶?2銆戯細鍓旈櫎 Batch 鍐呴噸澶嶇殑 User 鍜?Item锛岄槻姝⑩€滃亣闃存€р€濆悓璐ㄦ帓鏂?
            unique_users = torch.unique(user_id)
            unique_items = torch.unique(target_item_id)

            # 鍦ㄥ敮涓€鑺傜偣闆嗗悎涓婅绠?InfoNCE
            u_cl_loss = info_nce_loss(u1[unique_users], u2[unique_users], temp=self.config.temp)
            e_cl_loss = info_nce_loss(e1[unique_items], e2[unique_items], temp=self.config.temp)
            graph_cl_loss = u_cl_loss + e_cl_loss

        # ==========================================
        # 鎺╃爜鏇挎崲涓庡姩鎬佹帺鐮侀噸鏋勭洰鏍囨彁鍙?(DMR)
        # ==========================================
        dmr_loss = torch.tensor(0.0, device=self.device)
        if item_mask is not None and item_mask.any():
            target_mask_emb = seq_item_emb[item_mask].detach()
            seq_item_emb = seq_item_emb.clone()
            seq_item_emb[item_mask] = self.mask_token.to(seq_item_emb.dtype)

        user_time = target_time.unsqueeze(1)
        target_time_exp = target_time.unsqueeze(1)
        full_time = torch.cat([user_time, seq_time, target_time_exp], dim=1) / 3600.0

        gce_out, gce_seq_out = self.gce(stu_emb, seq_item_emb, seq_score, seq_item_id, e_disc_emb, full_time)

        # ==========================================
        # 璁＄畻 DMR InfoNCE 鎹熷け (搴熷純 MSE)
        # ==========================================
        if item_mask is not None and item_mask.any():
            masked_preds = gce_seq_out[item_mask]
            # 璁╄鎺╃爜棰勬祴鍑虹殑鐗瑰緛鍘讳笌鐪熷疄鐨勫浘鐗瑰緛寤虹珛姝ｆ牱鏈鍋跺叧绯?
            dmr_loss = info_nce_loss(masked_preds, target_mask_emb, temp=self.config.temp)

        # 浼犵粺 IRT 娴嬮噺瀛﹀榻愰娴嬪眰
        # 鍘熷瀹炵幇锛氱洿鎺ュ澶氱煡璇嗙偣棰樼洰鐨勭煡璇嗗悜閲忓仛姹傚拰锛屽鏄撹棰樼洰琛ㄧず闅忕煡璇嗙偣涓暟绾挎€ф斁澶с€?        # k_diff_emb = torch.mm(target_knowledge_emb, k_feature).to(self.device)
        knowledge_count = target_knowledge_emb.sum(dim=1, keepdim=True).clamp_min(1.0)
        k_diff_emb = torch.mm(target_knowledge_emb, k_feature).to(self.device) / knowledge_count
        stu_state = stu_emb + gce_out
        diff = stu_state - k_diff_emb
        pred = torch.sigmoid(self.irt_mlp(diff * e_disc_emb))

        # 杩斿洖涓婚娴嬨€侀噸鏋勫姣旀崯澶便€佸浘瀵规瘮鎹熷け
        return pred.squeeze(-1), dmr_loss, graph_cl_loss
