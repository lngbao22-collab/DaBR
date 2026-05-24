import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .model import Model
from numpy.random import RandomState

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class DaBR(Model):
    def __init__(self, config):
        super(DaBR, self).__init__(config)
        self.embedding_dim = 4 * self.config.hidden_size
        self.model_type = getattr(self.config, "model_type", "entity_relation_embedding")

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.embedding_dim)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.embedding_dim)
        self.rel_mask_embeddings = nn.Embedding(self.config.relTotal, self.embedding_dim)
        self.Dr = nn.Embedding(self.config.relTotal, self.embedding_dim)

        self.distance_lambda = float(getattr(self.config, "lmbda", 0.0))
        self.t_uni = float(getattr(self.config, "t_uni", 2.0))
        self.epsilon = float(getattr(self.config, "epsilon", 1e-8))
        self.gamma_uni_ent = float(getattr(self.config, "gamma_uni_ent", 1.0))
        self.gamma_uni_tail = float(getattr(self.config, "gamma_uni_tail", 1.0))
        self.gamma_uni_head = float(getattr(self.config, "gamma_uni_head", 1.0))
        self.gamma_uni_query = float(getattr(self.config, "gamma_uni_query", 1.0))
        self.gamma_neg = float(getattr(self.config, "gamma_neg", 1.0))
        self.eta1 = float(getattr(self.config, "eta1", getattr(self.config, "lmbda_two", 0.0)))
        self.eta2 = float(getattr(self.config, "eta2", getattr(self.config, "lmbda_two", 0.0)))
        self.criterion = nn.Softplus()

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_mask_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Dr.weight.data)

    @staticmethod
    def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
        norm = torch.sqrt(torch.sum(quaternion ** 2, 1, True)).clamp_min(1e-12)
        quaternion = quaternion / norm  # quaternion / norm
        quaternion = quaternion.reshape(-1, 4 * size)
        return quaternion

    @staticmethod
    def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
        i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
        j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
        k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
        return r2, i2, j2, k2

    @staticmethod
    def get_quaternion_wise_mul(quaternion):
        size = quaternion.size(1) // 4
        quaternion = quaternion.view(-1, 4, size)
        quaternion = torch.sum(quaternion, 1)
        return quaternion

    @staticmethod
    def vec_vec_wise_multiplication(q, p):  # vector * vector
        normalized_p = DaBR.normalization(p)  # bs x 4dim
        q_r, q_i, q_j, q_k = DaBR.make_wise_quaternion(q)  # bs x 4dim

        qp_r = DaBR.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
        qp_i = DaBR.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
        qp_j = DaBR.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
        qp_k = DaBR.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

    @staticmethod
    def get_inv(quaternion):
        q_r, q_i, q_j, q_k = torch.chunk(quaternion, 4, dim=1)
        quaternion_norm = q_r ** 2 + q_i ** 2 + q_j ** 2 + q_k ** 2
        quaternion_norm = torch.clamp(quaternion_norm, min=1e-12)
        r_inv = torch.cat([q_r / quaternion_norm, -q_i / quaternion_norm, -q_j / quaternion_norm, -q_k / quaternion_norm], dim=1)
        return r_inv

    @staticmethod
    def safe_uniformity(x, t_uni, epsilon):
        if x.size(0) < 2:
            return x.new_tensor(0.0)
        pairwise_distances = torch.pdist(x, p=2)
        return torch.log(torch.mean(torch.exp(-t_uni * pairwise_distances.pow(2))) + epsilon)

    def _positive_batch_size(self):
        return int(self.config.batch_size)

    def _unique_entities(self, heads, tails):
        return torch.unique(torch.cat([heads, tails], dim=0))

    def _unique_pair_ids(self, heads, relations):
        return torch.unique(heads * self.config.relTotal + relations)

    def _compose_entity_relation(self, h_ids, r_ids, t_ids):
        if self.model_type != "entity_relation_embedding":
            raise NotImplementedError(
                "semantic_embedding is not implemented in this repository; "
                "use entity_relation_embedding for the current DaBR pipeline."
            )

        h = self.normalization(self.ent_embeddings(h_ids))
        r = self.normalization(self.rel_embeddings(r_ids))
        t = self.normalization(self.ent_embeddings(t_ids))
        r_d = self.Dr(r_ids)
        h_mask = self.normalization(h * torch.sigmoid(self.rel_mask_embeddings(r_ids)))
        q = self.normalization(self.vec_vec_wise_multiplication(h_mask, r))
        t_target = self.normalization(self.vec_vec_wise_multiplication(t, self.get_inv(r)))
        return q, t_target, h_mask, t, r_d

    def _compose_query(self, h_ids, r_ids):
        if self.model_type != "entity_relation_embedding":
            raise NotImplementedError(
                "semantic_embedding is not implemented in this repository; "
                "use entity_relation_embedding for the current DaBR pipeline."
            )

        h = self.normalization(self.ent_embeddings(h_ids))
        r = self.normalization(self.rel_embeddings(r_ids))
        h_mask = self.normalization(h * torch.sigmoid(self.rel_mask_embeddings(r_ids)))
        q = self.normalization(self.vec_vec_wise_multiplication(h_mask, r))
        return q

    def _score(self, h_ids, r_ids, t_ids):
        q, t_target, h_mask, t, r_d = self._compose_entity_relation(h_ids, r_ids, t_ids)
        semantic_score = torch.sum(q * t_target, dim=-1)
        hrt = h_mask + r_d - t
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        distance_penalty = torch.norm(s_d + x_d + y_d + z_d, p=1, dim=-1)
        return distance_penalty - semantic_score

    @staticmethod
    def _calc(h, r, t, dr, para):
        #semantic
        hr = DaBR.vec_vec_wise_multiplication(h, r)
        r_inv = DaBR.get_inv(r)
        tr = DaBR.vec_vec_wise_multiplication(t, r_inv)
        score_s = hr * tr
        #distance
        hrt = h + dr - t
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        score_d = s_d + x_d + y_d + z_d
        return -torch.sum(score_s, -1) - para * torch.norm(score_d, p=1, dim=-1)

    @staticmethod
    def regularization(quaternion):  # vectorized quaternion bs x 4dim
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)


    def loss(self, score, regul1, regul2):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul1 + self.config.lmbda_two * regul2

    def forward(self):
        batch_size = self._positive_batch_size()
        h_ids = self.batch_h.long()
        t_ids = self.batch_t.long()
        r_ids = self.batch_r.long()
        batch_y = self.batch_y.float()

        score = self._score(h_ids, r_ids, t_ids)
        q, t_target, h_mask, t, r_d = self._compose_entity_relation(h_ids, r_ids, t_ids)

        q_pos = q[:batch_size]
        t_target_pos = t_target[:batch_size]
        h_mask_pos = h_mask[:batch_size]
        t_pos = t[:batch_size]
        r_d_pos = r_d[:batch_size]

        align_loss = torch.mean((q_pos - t_target_pos) ** 2)
        hrt = h_mask_pos + r_d_pos - t_pos
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        distance_penalty = torch.norm(s_d + x_d + y_d + z_d, p=1, dim=-1)
        align_loss = align_loss + self.distance_lambda * distance_penalty.mean()

        positive_h = h_ids[:batch_size]
        positive_t = t_ids[:batch_size]
        positive_r = r_ids[:batch_size]

        ent_ids = self._unique_entities(positive_h, positive_t)
        tail_ids = torch.unique(positive_t)
        head_ids = torch.unique(positive_h)
        query_ids = self._unique_pair_ids(positive_h, positive_r)

        uni_ent = self.safe_uniformity(
            self.normalization(self.ent_embeddings(ent_ids)), self.t_uni, self.epsilon
        )
        uni_tail = self.safe_uniformity(
            self.normalization(self.ent_embeddings(tail_ids)), self.t_uni, self.epsilon
        )
        uni_head = self.safe_uniformity(
            self.normalization(self.ent_embeddings(head_ids)), self.t_uni, self.epsilon
        )

        query_heads = torch.div(query_ids, self.config.relTotal, rounding_mode="floor")
        query_rels = torch.remainder(query_ids, self.config.relTotal)
        query_q = self._compose_query(query_heads, query_rels)
        uni_query = self.safe_uniformity(query_q, self.t_uni, self.epsilon)

        neg_loss = score.new_tensor(0.0)
        if self.gamma_neg > 0.0:
            neg_loss = torch.mean(F.softplus(batch_y * score))

        regularization = self.eta1 * torch.mean(self.ent_embeddings.weight ** 2)
        regularization = regularization + self.eta2 * (
            torch.mean(self.rel_embeddings.weight ** 2) + torch.mean(self.Dr.weight ** 2)
        )

        uni_total = (
            self.gamma_uni_ent * uni_ent
            + self.gamma_uni_tail * uni_tail
            + self.gamma_uni_head * uni_head
            + self.gamma_uni_query * uni_query
        )

        return align_loss + uni_total + self.gamma_neg * neg_loss + regularization

    def predict(self):
        score = self._score(self.batch_h.long(), self.batch_r.long(), self.batch_t.long())
        return score.cpu().data.numpy()
