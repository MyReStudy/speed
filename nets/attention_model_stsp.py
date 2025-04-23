import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from problems.stsp.graph_info import get_distance_and_neighbor, get_info_aisle_vertex
from utils.tensor_functions import compute_in_batches

from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from options import get_options

global distance_matrix
global neighbor


def set_distance_matrix(dm):
    global distance_matrix
    distance_matrix = dm


def set_neighbor(n):
    global neighbor
    neighbor = n


def get_distance_matrix():
    global distance_matrix
    return distance_matrix


def get_neighbor():
    global neighbor
    return neighbor


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_stsp = problem.NAME == 'stsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
        step_context_dim = embedding_dim + 1
        node_dim = 3  # x, y, demand / prize

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        from nets.graph_encoder import GraphAttentionEncoder

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            opt = get_options()
            if opt.problem == 'stsp':
                all_coords = input['loc']
                distance_matrix, neighbor = get_distance_and_neighbor(all_coords,
                                                                      input['required_len_after_delete'],
                                                                      input['steiner_len'],
                                                                      opt.aisle_num,
                                                                      opt.cross_num,
                                                                      opt.aisle_length,
                                                                      opt.cross_length)
                set_distance_matrix(distance_matrix)
                set_neighbor(neighbor)
                # embeddings, _ = self.embedder(self._init_embed(input))

            embeddings, _ = self.embedder(self._init_embed(input))  # 输入batch点的坐标 输出embeddings  5*14*128

        _log_p, pi, penalty = self._inner(input, embeddings)  # input就是bat bat就是按batch取出来的数 输出概率和选择的顺序

        cost, mask = self.problem.get_costs(input, pi, penalty)  # 这个stsp还没有
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        features = ('deterministic_prize',)

        return self.init_embed(torch.cat((  # 把loc每个坐标后面加上奖励后进行线性变化
            input['loc'],
            *(input[feat][:, :, None] for feat in features)
        ), -1))

    def _inner(self, input, embeddings):  # 输出每一步概率与最终结点序列

        outputs = []
        sequences = []

        state = self.problem.make_state(input)  # 初始化了特定问题

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (state.all_finished()):
            log_p, mask = self._get_log_p(fixed, state)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])  # 选择的概率
            sequences.append(selected)  # 选择的output的点

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state.cur_total_penalty

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # 图的embedding
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # 第一个维度和第三个维度保留，第二个维度设为1

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        # a = self._get_parallel_step_context(fixed.node_embeddings, state)
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        steienr_len = state.steiner_len
        have_picked = state.have_picked
        required_len = state.required_len
        ids = state.ids

        unvisited_points = []
        _, mask_consider_neighbor = state.get_mask()
        for i in ids:
            i = i.item()
            num_visited = len(have_picked[i])  # 已经访问过的必须访问点
            total_must_visit = required_len[i].item()
            current_node = state.prev_a[i][0].item()
            if num_visited >= 0.8 * total_must_visit and state.idx_vertex_info[i][current_node] == 's' \
                    and num_visited < total_must_visit:
                all_must_visit = torch.arange(steienr_len, total_must_visit + steienr_len)  # 所有必访点的index集合
                visited_set = set(have_picked[i])  # 访问过的必访点集合
                unvisited = [point for point in all_must_visit.tolist() if point not in visited_set]  # 获取目前还没访问的必访点
                unvisited_points.append(unvisited)
                unvisited_1 = [state.where_is_vertex[i][point] for point in unvisited]  # 他们所在的巷道
                unvisited_steiner_batch = []  # 巷道两端的Steiner点
                for a in unvisited_1:
                    s1 = state.aisle_vertex_info[i][a][0]
                    s2 = state.aisle_vertex_info[i][a][-1]
                    neighbor_for_s1 = state.neighbor_all[i][s1]
                    indices_s1 = torch.where(neighbor_for_s1[state.steiner_len:] == 1)[0] + state.steiner_len
                    all_in_order_visit_s1 = all(idx.item() in state.order_visit[i] for idx in indices_s1)
                    if not all_in_order_visit_s1:
                        unvisited_steiner_batch.append(s1)
                    neighbor_for_s2 = state.neighbor_all[i][s2]
                    indices_s2 = torch.where(neighbor_for_s2[state.steiner_len:] == 1)[0] + state.steiner_len
                    all_in_order_visit_s2 = all(idx.item() in state.order_visit[i] for idx in indices_s2)
                    if not all_in_order_visit_s2:
                        unvisited_steiner_batch.append(s2)
                unvisited_steiner_batch = list(set(unvisited_steiner_batch))
                neighbor_for_current_steiner = state.neighbor_all[i][current_node]
                # Step 1: 获取从 steiner_len 位置开始，值为1的下标
                indices = torch.where(neighbor_for_current_steiner[state.steiner_len:] == 1)[0] + state.steiner_len
                # Step 2: 判断是否所有下标都在 order_visit 中 是 TRUE 否FALSE
                all_in_order_visit = all(idx.item() in state.order_visit[i] for idx in indices)
                if all_in_order_visit:
                    mask_consider_neighbor[i] = state.get_mask_hou(mask_consider_neighbor[i], unvisited_steiner_batch)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask_consider_neighbor)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        return log_p, mask_consider_neighbor

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
        c = []
        for b in range(batch_size):
            em_for_batch = embeddings[b]
            cn_for_batch = current_node[b].item()
            c.append(em_for_batch[cn_for_batch].tolist())
        d = torch.tensor(c).view(batch_size, num_steps, embeddings.size(-1)).to(device=current_node.device)
        return torch.cat(
            (
                d,
                (
                    state.get_remaining_prize_to_collect()[:, :, None]
                )
            ),
            -1
        )

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size) 计算相容度
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(
            glimpse_Q.size(-1))  # 将 glimpse_K 张量的最后两个维度进行交换
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf  # 把mask的地方相容度设为无限大

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
