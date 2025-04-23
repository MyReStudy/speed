import math
import random
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle

from options import get_options
from problems.stsp.graph_info import delete_to_four_vertex_for_instance
# from problems.stsp.graph_info import delete_to_four_vertex
from problems.stsp.state_stsp import StateSTSP
from utils.beam_search import beam_search

class STSP(object):

    # random.seed(5)

    NAME = 'stsp'

    @staticmethod
    def _get_costs(dataset, pi, penalty):
        steiner_len = dataset['steiner_len'][0].item()
        shape = pi[:, :1].shape
        p_with_depot = torch.cat(
            (
                torch.full(shape,steiner_len,dtype=torch.int64, device=pi.device),
                pi
            ),
            1
        )
        loc_with_depot = dataset['all_loc_orig']
        d = loc_with_depot.gather(1, p_with_depot[..., None].expand(*p_with_depot.size(), loc_with_depot.size(-1)))

        length = (
            (d[:, 1:] - d[:, :-1]).abs().sum(dim=-1).sum(1)
        )

        return length, None  # 得到距离

    def make_dataset(*args, **kwargs):
        return STSPDataset(*args, **kwargs)

    @staticmethod
    def get_costs(dataset, pi, penalty):
        return STSP._get_costs(dataset, pi, penalty)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSTSP.initialize(*args, **kwargs, stochastic=False)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        # With beam search we always consider the deterministic case
        state = STSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)



def generate_steiner_points(aisle_num, cross_num, aisle_length, cross_length):
    # n = aisle_num*cross_num
    steiner_lst = []  # 用来存储所有的steiner点的坐标
    for i in range(0, cross_num):
        for j in range(0, aisle_num):
            steiner_lst.append((float(j*cross_length), float(i*aisle_length)))
    steiner_tensor = torch.tensor(steiner_lst)
    return steiner_tensor  # 返回tensor类型存储的Steiner点



def generate_instance(aisle_num, cross_num, aisle_length, cross_length):  # 生成一次的数据
    '''
    Args:
        size_required: 必须访问点的数量
        aisle_num: 巷道数量
        cross_num: 横向巷道数量
        aisle_length: 巷道长
        cross_length: 横向巷道长

    Returns:
    return {
        'depot': depot_loc, depot位置
        'loc': all_loc, 所有点的坐标
        'deterministic_prize': deterministic_prize 所有点的奖励
    }
    '''
    # random.seed(5)  # 如果加了这个每次生成的都是一样的instance
    # random_number = random.randint(0, 100)
    # np.random.seed(random_number)

    stenier_loc = generate_steiner_points(aisle_num, cross_num, aisle_length, cross_length)  # list steiner坐标
    depot_loc = stenier_loc[0]  # 默认的depot是仓库左下角
    required_loc = []

    for i in range(0, aisle_num * (cross_num - 1)):  # i代表巷道编号
        r = random.randint(0, 5)  # 产生长度相等的不同instance
        if i == 0:  # 默认depot在左下角
            if r==0 or r==1:  # 如果这个巷道里没存货物或者存着一个
                r = 1
            else:  # 否则 减去一个(之后还要把depot加进来)
                r = r-1
            # r=1
        y_list = random.sample(list(range((i // aisle_num) * aisle_length + 1,
                                          (1 + i // aisle_num) * aisle_length - 1)), r)  # 在第i个巷道范围内随机抽取r个点
        for j in range(0, len(y_list)):
            required_loc.append((float(i % aisle_num * (cross_length)), float(y_list[j])))  # 加进去x坐标

    coords = stenier_loc.numpy().tolist()+[depot_loc.numpy().tolist()]+required_loc  # 所有点 Steiner+depot+required
    steiner_len = len(stenier_loc)
    all_loc_orig, required_len_after_delete_with_depot, steiner_len = delete_to_four_vertex_for_instance(coords, steiner_len, aisle_num, cross_num)  # 进行一些节点的删减(最大gap的影响) 所有坐标包括depot

    # 分离x和y坐标
    x_coords = all_loc_orig[:, 0]
    y_coords = all_loc_orig[:, 1]

    # 计算x和y坐标的最大值和最小值
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    #
    if x_min<y_min:
        all_min = x_min
    else:
        all_min = y_min

    if x_max>y_max:
        all_max = x_max
    else:
        all_max = y_max

    # 归一化x坐标
    normalized_x_coords = (x_coords - all_min) / (all_max - all_min)
    # 归一化y坐标
    normalized_y_coords = (y_coords - all_min) / (all_max - all_min)
    # 合并归一化后的x和y坐标回tensor
    all_loc = torch.stack([normalized_x_coords, normalized_y_coords], dim=1)  # 归一化之后的所有坐标集合
    deterministic_prize = [0.0]*steiner_len+[1.0]*required_len_after_delete_with_depot  # prize信息 前面是Steiner点 后面是所有的必访点包括depot
    penalty = [1.0] * steiner_len+[0.0]*required_len_after_delete_with_depot
    return {
        'depot': all_loc[steiner_len],
        'loc': all_loc,  # 包括depot
        'deterministic_prize': torch.tensor(deterministic_prize),
        'penalty': torch.tensor(penalty),
        'all_loc_orig': all_loc_orig,
        'required_len_after_delete':required_len_after_delete_with_depot,
        'steiner_len':steiner_len,
        'coords_before_delete': torch.tensor(coords)
    }


class STSPDataset(Dataset):

    opt = get_options()

    def __init__(self, filename=None, size=6, aisle_num=opt.aisle_num, cross_num=opt.cross_num, aisle_length=opt.aisle_length, cross_length=opt.cross_length, num_samples=1000000, distribution=None):
        super(STSPDataset, self).__init__()

        self.data_set = []
        self.data = [
            generate_instance(aisle_num, cross_num, aisle_length, cross_length)
            for _ in range(num_samples)
        ]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":

    a = generate_steiner_points(2,3,5,2)
    b = generate_instance(6, 2,3,5,2)
    # vertexs_tensor = torch.cat((b['depot'], b['loc']), dim=1)
    data = [
        generate_instance(6, 2,3,5,2)
        for i in range(3)
    ]
    # c = get_distance_matrix(data,800,2,3)
    # e = get_idx_vertex_info(data)
    # f= e[0][2]
    d = 1