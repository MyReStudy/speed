import copy
import math
import random
from collections import deque

import numpy as np
import torch
from typing import NamedTuple
from options import get_options
from problems.stsp.graph_info import get_distance_and_neighbor, get_info_aisle_vertex
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F

class StateSTSP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    real_prize: torch.Tensor
    penalty: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    lengths: torch.Tensor
    cur_total_prize: torch.Tensor
    cur_total_penalty: torch.Tensor

    cur_coord: torch.Tensor  # 当前点的坐标
    i: torch.Tensor  # Keeps track of step
    mask: torch.Tensor
    prev_all: torch.Tensor  # 按顺序存储的访问顺序 # 每次真正output的点的集合，不包括中间经过的点
    order_visit:list  # 列表形式的访问顺序？真正output的点和中间经过的点都计入
    required_len:torch.Tensor  # 必须访问点长度不包括depot
    steiner_len:int  # Steiner点长度
    last_chosen_for_steiner:list  # 上一个选择的点的idx
    neighbor_all:list
    distance_matrix:list
    aisle_vertex_info:list  # 每个巷道里存着哪些拣货点
    where_is_vertex: list  # steiner点下标取值是-1 必访点取值是他们所在的巷道
    idx_vertex_info:list  # 每个点是Steiner点还是必访点
    have_picked: list  # 已经拿到的货物点 访问过的必须访问点


    # @property
    # def visited(self):
    #     if self.visited_.dtype == torch.uint8:
    #         return self.visited_
    #     else:
    #         return mask_long2bool(self.visited_, n=self.coords.size(-2))

    def __getitem__(self, key):
        # assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            lengths=self.lengths[key],
            cur_total_prize=self.cur_total_prize[key],
            cur_total_penalty=self.cur_total_penalty[key],
            cur_coord=self.cur_coord[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8, stochastic=False):
        depot = input['depot']
        loc = input['loc']
        required_len = torch.sum(input['deterministic_prize'] == 1, dim=1, keepdim=True)
        steiner_len = torch.sum(input['deterministic_prize'][0] == 0).item()
        real_prize = input['deterministic_prize']

        batch_size, n_loc, _ = loc.size()
        coords = loc
        # real_prize_with_depot = real_prize

        opt = get_options()
        aisle_length, cross_length, aisle_num, cross_num, cross_length = opt.aisle_length, opt.cross_length, opt.aisle_num, opt.cross_num, opt.cross_length
        require_len_for_batch = required_len.squeeze(1).tolist()
        steiner_len_for_batch = [steiner_len for _ in range(len(require_len_for_batch))]
        distance_matrix, neighbor_all = get_distance_and_neighbor(coords, require_len_for_batch, steiner_len_for_batch, aisle_num, cross_num, aisle_length, cross_length)
        aisle_vertex_info, where_is_vertex = get_info_aisle_vertex(coords, require_len_for_batch, steiner_len_for_batch, aisle_num, cross_num)  # 所有batch每个巷道内点idx，所有batch每个点存在哪个巷道
        penalty = input['penalty']

        idx_vertex_info=[]
        have_picked=[]
        for batch in range(coords.size(0)):
            have_picked.append([steiner_len])  # 从起止点出发
            vs_dict = {}
            r = required_len[batch]
            for i in range(steiner_len):
                vs_dict[i] = 's'
            for i in range(steiner_len, 1 + r + steiner_len):
                vs_dict[i] = 'v'
            idx_vertex_info.append(vs_dict)
        # 初始化禁忌列表
        last_chosen_for_steiner = [[] for _ in range(coords.size(0))]
        for b in range(len(last_chosen_for_steiner)):
            for i in range(cross_num):
                for j in range(aisle_num):
                    if i==0 or i==cross_num-1 or j==0 or j==aisle_num-1:
                        if (i==0 and j==0) or (i==0 and j==aisle_num-1) or (i==cross_num-1 and j==0) or (i==cross_num-1 and j==aisle_num-1):
                            last_chosen_for_steiner[b].append(deque(maxlen=1))
                        else:
                            last_chosen_for_steiner[b].append(deque(maxlen=2))
                    else:
                        last_chosen_for_steiner[b].append(deque(maxlen=3))

        return StateSTSP(
            coords=coords,  # 坐标
            real_prize=real_prize,  # 感觉和expected_prize没区别
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.full((batch_size, 1), steiner_len, dtype=torch.long, device=loc.device),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_total_prize=torch.ones(batch_size, 1, device=loc.device),  # 从depot出发 已经得到了一个prize
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            mask=torch.zeros(coords.shape[0], 1, coords.shape[1]),
            order_visit = [[steiner_len] for i in range(coords.shape[0])],  # 中间的点也加进去的拣货路径
            prev_all=torch.tensor([[steiner_len] for _ in range(batch_size)], dtype=torch.long, device=loc.device),
            required_len = required_len,
            steiner_len = steiner_len,
            last_chosen_for_steiner = last_chosen_for_steiner,
            neighbor_all=neighbor_all,
            distance_matrix=distance_matrix,
            aisle_vertex_info=aisle_vertex_info,
            where_is_vertex=where_is_vertex,
            idx_vertex_info=idx_vertex_info,
            penalty=penalty,
            cur_total_penalty=torch.zeros(batch_size, 1, device=loc.device),
            have_picked=have_picked
        )

    def get_remaining_prize_to_collect(self):
        # returns the remaining prize to collect, or 0 if already collected the minimum (1.0)
        a=self.required_len - self.cur_total_prize
        return torch.clamp(self.required_len - self.cur_total_prize, min=0)  # 如果小于0依旧是0

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.
        # We are at the depot so no need to add remaining distance
        return self.lengths
        # return self.lengths + self.cur_total_penalty

    def update(self, selected):
        # 更新距离 更新访问点序列 更新prize
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step # 上一个选择的点

        aisle_vertex_info = self.aisle_vertex_info
        where_is_vertex = self.where_is_vertex
        distance_matrix, neighbor = self.distance_matrix, self.neighbor_all


        selected_coord = self.coords[self.ids, selected].squeeze(1)  # 当前选择点的坐标
        prev_coord = self.coords[self.ids, self.prev_a].squeeze(1)
        distance_between_selected_prev = torch.sum(torch.abs(selected_coord - prev_coord), dim=1)
        selected_idx_list = selected.tolist()  # 这次选择的点
        prev_idx_list = self.prev_a.tolist()  # 上一轮选择的点

        lengths = []
        idx_vertex_info=self.idx_vertex_info
        last_chosen_for_steiner=self.last_chosen_for_steiner
        order_visit = self.order_visit
        prev_a = self.prev_a
        cur_coord = self.cur_coord

        for i in range(0, len(selected_idx_list)):
            lengths.append(self.lengths[i].item() + distance_between_selected_prev[i].item())  # 加上距离
            select = selected_idx_list[i][0]
            prev_a[i]=select
            cur_coord[i] = selected_coord[i]
            prev = prev_idx_list[i][0]
            if idx_vertex_info[i][prev]=='s':
                last_chosen_for_steiner[i][prev].append(select)

            # 补全实际的情况
            if select!=self.steiner_len :
                if idx_vertex_info[i][select]=='v' or idx_vertex_info[i][prev]=='v':  # 只有两个点都是必访点才会有这种中间塞一堆点的情况
                    select_aisle = where_is_vertex[i][select] if idx_vertex_info[i][select]=='v' else where_is_vertex[i][prev]  # 获取选择的点所在的巷道
                    all_vertex_in_select_aisle = aisle_vertex_info[i][select_aisle]  # 获取当前巷道内所有的点
                    try:
                        select_idx = all_vertex_in_select_aisle.index(select)  # 这轮选择的点在哪里
                    except:
                        a=1
                    prev_idx = all_vertex_in_select_aisle.index(prev)  # 上轮选择的点在哪里
                    if select_idx<prev_idx:  # 目的：选出来上轮选择的点到当前访问的点，经过了多少别的点
                        middle = all_vertex_in_select_aisle[select_idx+1:prev_idx]
                        middle = middle[::-1]
                    else:
                        middle = all_vertex_in_select_aisle[prev_idx+1:select_idx]

                    # 下面更新prize
                    for m in middle:
                        if m not in order_visit[i]:
                            value_to_add = self.real_prize[i, m].item()
                            self.cur_total_prize[i] += value_to_add
                        order_visit[i].append(m)
                        self.have_picked[i].append(m)
                        self.mask[i][0][m] = 1

            if select not in order_visit[i]:
                value_to_add = self.real_prize[i, select].item()
                self.cur_total_prize[i] += value_to_add
                if idx_vertex_info[i][select]=='v':
                    self.have_picked[i].append(select)

            if idx_vertex_info[i][select]=='s':
                self.cur_total_penalty[i]+=self.penalty[i,select].item()

            # 更新真实访问顺序列表
            order_visit[i].append(select)  # 将中间经过的点和当前访问的点加到order里

        # 转化长度格式
        lengths = torch.tensor(lengths)
        cur_total_prize = self.cur_total_prize
        cur_total_penalty = self.cur_total_penalty
        have_picked = self.have_picked

        return self._replace(
            prev_a=prev_a,
            # visited_=visited_,
            lengths=lengths,
            cur_total_prize=cur_total_prize,
            cur_total_penalty=cur_total_penalty,
            cur_coord=cur_coord,
            prev_all=torch.cat((self.prev_all, prev_a), dim=1),  # 新增prev_all 代表按顺序访问的所有点 每次有了新点加到后面去
            i=self.i + 1,
            last_chosen_for_steiner = last_chosen_for_steiner,
            have_picked=have_picked
            # edge_times=edge_times
        )

    def all_finished(self):
        return self.i.item() > 0 and (self.prev_a == self.steiner_len).all()  # 不是第一轮迭代并且最近一个访问的都是0

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask_hou(self, mask_consider_neighbor_i, not_steiner_list):
        # 当大于一定阈值之后，取出来没有被访问的必须访问点两端的Steiner点进行转移
        if torch.cuda.is_available():
            device = torch.device('cuda:0')  # 使用第一个GPU
        else:
            device = torch.device('cpu')  # 使用CPU

        mask_consider_neighbor_i[0,:] = 1
        mask_consider_neighbor_i[0,not_steiner_list] = 0

        return mask_consider_neighbor_i.to(device)


    def get_mask(self):
        '''
        这是改写的self.mask规则 self.mask最后返回的是一堆true false
        order_visit(batch*可变list)是访问的顺序 按照顺序存储每一个节点的下标
        list (batch*(depot+vertex_num))这是统一记录下标与节点对应的list
        prev_a 最近访问的点
        '''
        if torch.cuda.is_available():
            device = torch.device('cuda:0')  # 使用第一个GPU
        else:
            device = torch.device('cpu')  # 使用CPU

        steiner_len = self.steiner_len
        idx_vertex_info=self.idx_vertex_info
        require_len_for_batch = self.required_len.squeeze(1).tolist()
        steiner_len_for_batch = [steiner_len for _ in range(len(require_len_for_batch))]
        neighbor_all=self.neighbor_all
        order_visit = self.order_visit

        last_visit = [row[-1] for row in order_visit]  # 取出最后一个元素

        for i in range(len(last_visit)):  # 对每一个batch进行遍历
            not_padding_num= require_len_for_batch[i]+steiner_len_for_batch[i]
            self.mask[i][0][not_padding_num:] = 1  # 将padding来的都设为1 被mask

            if idx_vertex_info[i][last_visit[i]] == 'v':  # 如果这个点是必访点，那么一定会遮上
                self.mask[i][0][last_visit[i]] = 1
            else:
                if len(order_visit[i])>4:
                    previous_vertex = order_visit[i][-5:-2]  # 取出来到这个Steiner点之前的四个点
                else:
                    previous_vertex = order_visit[i][:-1]
                for v in previous_vertex:
                    if idx_vertex_info[i][v] == 'v':
                        self.mask[i][0][v] = 1

        mask_consider_neighbor = copy.deepcopy(self.mask)

        # 把不是邻居的点遮上
        for b in range(0, len(last_visit)):
            neighbor = neighbor_all[b]
            last_visit_for_batch = last_visit[b]
            neighbor = neighbor[last_visit_for_batch]
            for i in range(0, len(neighbor)):
                tmp = int(neighbor[i].item())
                if tmp != 1:
                    mask_consider_neighbor[b][0][i] = 1

        # 这个地方改成只对steiner点记录，有一个队列，包含三个元素，满了就把最前面的挤走
        last_chosen_for_steiner = self.last_chosen_for_steiner
        for b in self.ids:  # 对batch进行遍历
            i = b.item()
            last_visit_for_batch = last_visit[i]
            if idx_vertex_info[i][last_visit_for_batch] == 's':
                pre_chosen_for_batch_steiner = last_chosen_for_steiner[i][last_visit_for_batch]
                for item in pre_chosen_for_batch_steiner:
                    mask_consider_neighbor[i][0][item] = 1

        mask_consider_neighbor = mask_consider_neighbor.to(torch.uint8)
        all_mask = mask_consider_neighbor.squeeze(1).all(dim=-1)  # 看是不是都被mask上了
        for i in range(len(all_mask)):  # 对batch进行遍历
            if all_mask[i].item() != 0:
                neighbor = neighbor_all[i]  # 取出那个batch的邻接矩阵
                last_visit_for_batch = last_visit[i]  # 取出来最后一个访问的元素
                neighbor = neighbor[last_visit_for_batch]  # 取出来最后一个访问的元素的邻居
                for j in range(0, len(neighbor)):
                    tmp = int(neighbor[j].item())
                    if tmp == -1 and j not in last_chosen_for_steiner[i][last_visit_for_batch]:  # 如果是一个巷道两端的steiner点
                        mask_consider_neighbor[i][0][j] = 0  # 把mask解开

        # 如果都访问完了 对非depot点进行mask
        for b in self.ids:  # 对batch进行遍历
            i = b.item()
            if self.cur_total_prize[i].item() >= require_len_for_batch[i] and idx_vertex_info[i][last_visit[i]] == 's'\
            or (self.cur_total_prize[i].item() >= require_len_for_batch[i] and last_visit[i] == steiner_len):
                self.mask[i,:,steiner_len]=0
                self.mask[i,:,:steiner_len]=1
                self.mask[i,:,steiner_len+1:]=1
                mask_consider_neighbor[i, :, steiner_len] = 0
                mask_consider_neighbor[i, :, :steiner_len] = 1
                mask_consider_neighbor[i, :, steiner_len + 1:] = 1

        return self.mask.to(device) > 0, mask_consider_neighbor.to(device) > 0


    def construct_solutions(self, actions):
        return actions
