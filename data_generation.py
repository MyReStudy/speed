import random
import torch.nn.functional as F
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_data(aisle_num, cross_num, require_info):
    orders = [0]
    unique_aisles = np.unique(require_info[3])
    for aisle in unique_aisles:
        indices = np.where(require_info[3] == aisle)[0]
        values = require_info[7][indices]
        num_samples = random.randint(0, 5)
        if aisle == 1:  # 默认depot在左下角
            if num_samples==0 or num_samples==1:  # 如果这个巷道里没存货物或者存着一个
                num_samples = 1
            else:  # 否则 减去一个(之后还要把depot加进来)
                num_samples = num_samples-1
        sampled_values = random.sample(list(values), num_samples)
        orders=orders+sampled_values

    orders_info = require_info[:, orders]  # 把这些点对应的信息取出来
    group_keys = orders_info[3]  # 取出来所在巷道
    group_values = orders_info[7]  # 取出来他们的index
    sorted_indices = np.argsort(group_values)  # 按 require_info[7] 升序排列 返回索引
    group_keys_sorted = group_keys[sorted_indices]  # 获取排序后所在巷道
    group_values_sorted = group_values[sorted_indices]  # 获取排序后的index orders_info[7]升序排序
    # sorted_indices_exact = np.insert(sorted_indices, 0, 0)
    depot = [require_info[6][0]]
    all_loc_before_delete = depot+orders_info[6].tolist()  # 排序后的坐标

    # print(all_loc_before_delete)

    _, group_start = np.unique(group_keys_sorted, return_index=True)  # 获取unique的巷道在group_keys_sorted开始的位置 每个巷道开始的位置
    group_sizes = np.diff(np.append(group_start, len(group_keys_sorted)))  # 看看每个巷道里有几个点

    order_after_delete = np.array([])  # 存最终删减完的数据

    neighbor_nodes = {}
    for i, (start, size) in enumerate(zip(group_start, group_sizes)):
        if i == len(group_start) - 1:  # i是最后一个
            goods_in_aisle = group_values_sorted[group_start[i]:]  # 取出来这个巷道里所有的货物
        else:
            goods_in_aisle = group_values_sorted[group_start[i]:group_start[i + 1]]
        if size > 3:
            min_index = np.argmin(goods_in_aisle)  # 最小的idx
            max_index = np.argmax(goods_in_aisle)  # 最大的idx
            diffs = np.abs(np.diff(goods_in_aisle))  # 巷道里不同idx的差值
            max_diff_idx = np.argmax(diffs)  # 最大的差值所在的位置
            max_diff_indices = [max_diff_idx, max_diff_idx + 1]  # 最大差值的两个idx
            result_indices = np.unique([min_index, max_index] + max_diff_indices)
            order_after_delete = np.append(order_after_delete, goods_in_aisle[result_indices]).astype(int)  # 把删完的点加进去
            if (goods_in_aisle[min_index]!=goods_in_aisle[max_diff_idx]):
                    # and (goods_in_aisle[min_index]!=0 and goods_in_aisle[max_diff_idx]!=0)):
                if goods_in_aisle[min_index]==0:
                    neighbor_nodes[goods_in_aisle[min_index]] = goods_in_aisle[max_diff_idx]
                    # pass
                else:
                    neighbor_nodes[goods_in_aisle[min_index]] = goods_in_aisle[max_diff_idx]
                    neighbor_nodes[goods_in_aisle[max_diff_idx]] = goods_in_aisle[min_index]
            if goods_in_aisle[max_index]!=goods_in_aisle[max_diff_idx+1]:
                neighbor_nodes[goods_in_aisle[max_index]] = goods_in_aisle[max_diff_idx+1]
                neighbor_nodes[goods_in_aisle[max_diff_idx+1]] = goods_in_aisle[max_index]
        else:
            order_after_delete = np.append(order_after_delete, goods_in_aisle).astype(int)

    order_after_delete = np.insert(order_after_delete,0,0)
    # locations = np.argwhere(require_info[7]==order_after_delete)
    orders_coords_std = require_info[6][order_after_delete]  # 把这些点对应的坐标取出来
    orders_coords_std = torch.tensor(orders_coords_std.tolist())
    order_after_delete = torch.tensor(order_after_delete.tolist())
    orders_coords_std.to(device)
    order_after_delete.to(device)

    coords_orig = orders_coords_std
    padding_length = 4 * aisle_num * (cross_num-1)+1 - len(orders_coords_std)
    orders_coords_std = F.pad(orders_coords_std, (0, 0, 0, padding_length), "constant", 0)
    order_after_delete = F.pad(order_after_delete, (0, padding_length), "constant", 0)

    # time6 = time.time()
    # print(f'生成数据耗时{time6-time3}')
    return {'order_after_delete': order_after_delete,
            'orders_coords_std': orders_coords_std,
            'coords_orig':coords_orig,
            'neighbor_nodes':neighbor_nodes,
            'all_loc_before_delete':all_loc_before_delete,
            'orders_idx':orders}

