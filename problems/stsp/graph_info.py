import torch
from torch.nn.utils.rnn import pad_sequence


def data_padding(batch_data):
    loc_list = []
    prize_list = []
    orig_loc_list = []
    penalty_list = []
    for d in batch_data:
        loc_list.append(d['loc'])
        prize_list.append(d['deterministic_prize'])
        orig_loc_list.append(d['all_loc_orig'])
        penalty_list.append(d['penalty'])
    loc_pad = pad_sequence(loc_list, batch_first=True, padding_value=-1)
    prize_pad = pad_sequence(prize_list, batch_first=True, padding_value=-1)
    orig_loc_pad = pad_sequence(orig_loc_list, batch_first=True, padding_value=-1)
    penalty_pad = pad_sequence(penalty_list, batch_first=True, padding_value=-1)
    for i in range(len(batch_data)):
        batch_data[i]['loc'] = loc_pad[i]
        batch_data[i]['deterministic_prize'] = prize_pad[i]
        batch_data[i]['all_loc_orig'] = orig_loc_pad[i]
        batch_data[i]['penalty'] = penalty_pad[i]
    return batch_data


def delete_to_four_vertex_for_instance(coords, steiner_len, aisle_num, cross_num):
    '''
    Args:
        coords:
        steiner_len:
        aisle_num:
        cross_num:

    Returns:

    '''

    steiner_loc = coords[:steiner_len]  # 所有的Steiner点
    require_loc_after_delete = []  # 用于存batch内每个instance删去不需要的点之后的必须访问点
    middle_dict = {}
    points_dict_x = {}

    for point_index, (x, y) in enumerate(coords):  # 遍历所有
        x_val = x  # x的取值
        if x_val in points_dict_x:
            points_dict_x[x_val].append((point_index, y))  # [x]: 点的序号, y
        else:
            points_dict_x[x_val] = [(point_index, y)]  # points_dict_x[x坐标] = [(点的idx, y坐标)]
    # 对字典中每个键对应的值按照元组的第二个元素升序排列
    sorted_dict = {key: sorted(value, key=lambda x: x[1]) for key, value in points_dict_x.items()}  # x取值相同，y升序排列
    # 将排序后每个键对应值的元组的第一个元素取出来，以列表形式存储
    first_elements = {key: [item[0] for item in value] for key, value in sorted_dict.items()}  # 对应的点的idx

    # 对所有竖直巷道的steiner点做neighbor
    for j in range(0, aisle_num * (cross_num - 1)):  # 遍历所有的巷道
        bottom_steiner = j  # 巷道下面的steiner点
        top_steiner = j + aisle_num  # 巷道上面的Steiner点
        x_axis = coords[bottom_steiner][0]  # 获取下面steiner点的x坐标
        vertex_in_aisle = first_elements[x_axis]  # 把巷道内所有的点取出来
        bottom_steiner_idx = vertex_in_aisle.index(bottom_steiner)  # 下面的Steiner点的idx 只是在vertex_in_aisle中
        top_steiner_idx = vertex_in_aisle.index(top_steiner)  # 上面的Steiner点的Idx 只是在vertex_in_aisle中
        require_in_aisle = vertex_in_aisle[bottom_steiner_idx + 1:top_steiner_idx]  # 巷道内所有必须访问点的idx 在整个coords中
        if require_in_aisle:  # 如果有必须访问的点 先加第一个点 再加最大gap两端的点 最后加最后一个点
            four_friends = [require_in_aisle[0]]  # 先把第一个点弄进来
            gap = 0
            gap_v1 = require_in_aisle[0]
            gap_v2 = require_in_aisle[-1]
            if len(require_in_aisle) >= 2:  # 如果这个巷道里存的点多于2
                for v in range(len(require_in_aisle) - 1):
                    y1 = coords[require_in_aisle[v]][1]
                    y2 = coords[require_in_aisle[v + 1]][1]
                    tmp = abs(y1 - y2)
                    if tmp > gap:
                        gap_v1 = require_in_aisle[v]  # y坐标小的那个
                        gap_v2 = require_in_aisle[v + 1]
                        gap = tmp
            if gap_v1 not in four_friends:
                four_friends.append(gap_v1)
            if gap_v2 not in four_friends:
                four_friends.append(gap_v2)
            if require_in_aisle[-1] not in four_friends:
                four_friends.append(require_in_aisle[-1])  # 这是一个巷道里的四个点，按照y坐标由小到大排列

            for x in four_friends:
                require_loc_after_delete.append(coords[x])

            for t in range(len(four_friends) - 1):  # 把这两个点之间省略的那些点加进来 为了之后生成路径
                x1 = four_friends[t]  # require点的下标
                x2 = four_friends[t + 1]
                idx1 = require_in_aisle.index(x1)
                idx2 = require_in_aisle.index(x2)
                if abs(idx1 - idx2) != 1:
                    middle_dict[(x1, x2)] = require_in_aisle[idx1 + 1:idx2]
                    middle_dict[(x2, x1)] = require_in_aisle[idx1 + 1:idx2][::-1]
    all_loc_new = steiner_loc+require_loc_after_delete  # 所有更新后的点的坐标 包括depot
    required_len_after_delete = len(require_loc_after_delete)  # 删减后的必访点长度 包括depot点
    all_loc_new = torch.tensor(all_loc_new)
    return all_loc_new, required_len_after_delete, steiner_len

def delete_to_four_vertex_for_batch(coords_all, require_len, aisle_num, cross_num):  # 针对一个batch 即coords维度为5*13*2
    '''
    Args:
        coords_all: 所有的坐标集合 包括depot
        require_len: 必须要访问点的长度 包括depot
        steiner_len: Steiner点的长度
        aisle_num: picking aisle数量
        cross_num: cross aisle数量

    Returns: 删除完点的坐标集合,每个批次里必须访问点和Steiner点的数量是多少,所有涵盖的中间点

    '''
    coords_all_after_delete = []
    middle_points = []
    aisle_dict = {}
    require_len_for_batch = []  # 这里记录一个batch里不同instance包含必须访问点的个数
    steiner_len_for_batch = []  # 记录一个batch里不同instance包含Steiner点的个数

    for i in range(coords_all.size(0)):
        coords = coords_all[i].tolist()
        steiner_loc = coords[require_len:]  # 所有的Steiner点

        require_loc_after_delete = []  # 用于存batch内每个instance删去不需要的点之后的必须访问点
        steiner_len_for_batch.append(len(steiner_loc))

        middle_dict = {}
        points_dict_x = {}

        for point_index, (x, y) in enumerate(coords):  # 遍历所有
            x_val = x  # x的取值
            if x_val in points_dict_x:
                points_dict_x[x_val].append((point_index, y))  # [x]: 点的序号, y
            else:
                points_dict_x[x_val] = [(point_index, y)]  # points_dict_x[x坐标] = [(点的idx, y坐标)]
        # 对字典中每个键对应的值按照元组的第二个元素升序排列
        sorted_dict = {key: sorted(value, key=lambda x: x[1]) for key, value in points_dict_x.items()}  # x取值相同，y升序排列
        # 将排序后每个键对应值的元组的第一个元素取出来，以列表形式存储
        first_elements = {key: [item[0] for item in value] for key, value in sorted_dict.items()}  # 对应的点的idx

        # 对所有竖直巷道的steiner点做neighbor
        for j in range(0, aisle_num * (cross_num - 1)):  # 遍历所有的巷道
            aisle_dict[j]=[]
            bottom_steiner = require_len + j  # 巷道下面的steiner点
            top_steiner = require_len + j + aisle_num  # 巷道上面的Steiner点
            x_axis = coords[bottom_steiner][0]  # 获取下面steiner点的x坐标
            vertex_in_aisle = first_elements[x_axis]  # 把巷道内所有的点取出来
            bottom_steiner_idx = vertex_in_aisle.index(bottom_steiner)  # 下面的Steiner点的idx 只是在vertex_in_aisle中
            top_steiner_idx = vertex_in_aisle.index(top_steiner)  # 上面的Steiner点的Idx 只是在vertex_in_aisle中
            require_in_aisle = vertex_in_aisle[bottom_steiner_idx+1:top_steiner_idx]  # 巷道内所有必须访问点的idx 在整个coords中
            if require_in_aisle:  # 如果有必须访问的点
                four_friends = [require_in_aisle[0]]  # 先把第一个点弄进来
                gap = 0
                gap_v1 = require_in_aisle[0]
                gap_v2 = require_in_aisle[-1]
                if len(require_in_aisle)>=2:
                    for v in range(len(require_in_aisle)-1):
                        y1 = coords[require_in_aisle[v]][1]
                        y2 = coords[require_in_aisle[v+1]][1]
                        tmp = abs(y1-y2)
                        if tmp>gap:
                            gap_v1 = require_in_aisle[v]  # y坐标小的那个
                            gap_v2 = require_in_aisle[v+1]
                            gap = tmp
                if gap_v1 not in four_friends:
                    four_friends.append(gap_v1)
                if gap_v2 not in four_friends:
                    four_friends.append(gap_v2)
                if require_in_aisle[-1] not in four_friends:
                    four_friends.append(require_in_aisle[-1])  # 这是一个巷道里的四个点，按照y坐标由小到大排列

                for x in four_friends:
                    require_loc_after_delete.append(coords[x])

                for t in range(len(four_friends)-1):  # 把这两个点之间省略的那些点加进来
                    x1 = four_friends[t]  # require点的下标
                    x2 = four_friends[t+1]
                    idx1 = require_in_aisle.index(x1)
                    idx2 = require_in_aisle.index(x2)
                    if abs(idx1-idx2)!=1:
                        middle_dict[(x1, x2)] = require_in_aisle[idx1+1:idx2]
                        middle_dict[(x2, x1)] = require_in_aisle[idx1+1:idx2][::-1]
        all_loc_new = require_loc_after_delete+steiner_loc
        all_loc_new = torch.tensor(all_loc_new)
        coords_all_after_delete.append(all_loc_new)
        middle_points.append(middle_dict)
        require_len_for_batch.append(len(require_loc_after_delete))

    padded_sentences = pad_sequence(coords_all_after_delete, batch_first=True, padding_value=-1)
    # coords_all_after_delete = torch.stack(coords_all_after_delete, dim=0)
    return padded_sentences, require_len_for_batch, steiner_len_for_batch, middle_points

def get_distance_and_neighbor(coords_all_after_delete, require_len_for_batch, steiner_len_for_batch, aisle_num, cross_num, aisle_length, cross_length):

    distance_matrix = []
    neighbor = []
    require_len = require_len_for_batch

    for i in range(coords_all_after_delete.size(0)):  # 对于每个batch分别计算
        required_len_for_instancei = require_len[i]  # 这个instance的必访点数量
        steiner_len_for_steineri = steiner_len_for_batch[i]  # 这个instance Steiner点数量 为了不对后面padding的内容进行距离矩阵和邻居的计算
        coords = coords_all_after_delete[i].tolist()[0:required_len_for_instancei+steiner_len_for_steineri]  # 没有那些padding的数
        v_num = len(coords)
        distance_matrix_tmp = torch.full((v_num, v_num), float('inf'))
        neighbor_tmp = torch.full((v_num, v_num), float(0))
        points_dict_x = {}
        for point_index, (x, y) in enumerate(coords):
            x_val = x  # x的取值
            if x_val in points_dict_x:
                points_dict_x[x_val].append((point_index, y))  # [x]: 点的序号, y
            else:
                points_dict_x[x_val] = [(point_index, y)]  # points_dict_x[x坐标] = [(点的idx, y坐标)]
        # 对字典中每个键对应的值按照元组的第二个元素升序排列
        sorted_dict = {key: sorted(value, key=lambda x: x[1]) for key, value in points_dict_x.items()}  # x取值相同，y升序排列
        # 将排序后每个键对应值的元组的第一个元素取出来，以列表形式存储
        first_elements = {key: [item[0] for item in value] for key, value in sorted_dict.items()}  # 对应的点的idx
        for j in range(0, aisle_num * (cross_num - 1)):  # 遍历所有的巷道
            bottom_steiner = j  # 巷道下面的steiner点
            top_steiner = j + aisle_num  # 巷道上面的Steiner点
            distance_matrix_tmp[top_steiner, bottom_steiner] = aisle_length  # 先储存Steiner点之间的信息
            distance_matrix_tmp[bottom_steiner, top_steiner] = aisle_length

            x_axis = coords[bottom_steiner][0]  # 获取下面steiner点的x坐标
            vertex_in_aisle = first_elements[x_axis]  # 把巷道内所有的点取出来
            bottom_steiner_idx = vertex_in_aisle.index(bottom_steiner)  # 下面的Steiner点的idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            top_steiner_idx = vertex_in_aisle.index(top_steiner)  # 上面的Steiner点的Idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            require_in_aisle = vertex_in_aisle[bottom_steiner_idx+1:top_steiner_idx]  # 巷道内所有必须访问点的idx

            if j==0:  # 第一个巷道 depot要和上下Steiner点相连
                distance_matrix_tmp[bottom_steiner, require_in_aisle[0]] = abs(
                    sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                distance_matrix_tmp[top_steiner, require_in_aisle[0]] = abs(
                    sum([coords[top_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                neighbor_tmp[top_steiner, require_in_aisle[0]] = 1
                neighbor_tmp[bottom_steiner, require_in_aisle[0]] = 1

            if require_in_aisle:  # 如果有必须访问的点
                if len(require_in_aisle)==4:
                    # 对于最下面的Steiner点
                    distance_matrix_tmp[bottom_steiner, require_in_aisle[1]] = abs(sum([coords[bottom_steiner][j]-coords[require_in_aisle[1]][j] for j in range(2)]))
                    neighbor_tmp[bottom_steiner, require_in_aisle[1]] = 1

                    # 对于最下面的必访点
                    distance_matrix_tmp[require_in_aisle[0], bottom_steiner] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    distance_matrix_tmp[require_in_aisle[0], top_steiner] = abs(sum([coords[top_steiner][j]-coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[0], top_steiner] = 1
                    neighbor_tmp[require_in_aisle[0], bottom_steiner] = 1

                    # 对于下回撤点
                    distance_matrix_tmp[require_in_aisle[1], require_in_aisle[3]] = abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[3]][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], require_in_aisle[3]] = 1
                    distance_matrix_tmp[require_in_aisle[1], bottom_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[bottom_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], bottom_steiner] = 1
                    distance_matrix_tmp[require_in_aisle[1], top_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], top_steiner] = 1

                    # 对于上回撤点
                    distance_matrix_tmp[require_in_aisle[2], top_steiner] = abs(sum([coords[require_in_aisle[2]][j]-coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[2], top_steiner] = 1
                    distance_matrix_tmp[require_in_aisle[2], require_in_aisle[0]] = abs(sum([coords[require_in_aisle[2]][j]-coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[2], require_in_aisle[0]] = 1
                    distance_matrix_tmp[require_in_aisle[2], bottom_steiner] = abs(sum([coords[require_in_aisle[2]][j]-coords[bottom_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[2], bottom_steiner] = 1

                    # 对于最上面的必访点
                    distance_matrix_tmp[require_in_aisle[3], top_steiner] = abs(sum([coords[require_in_aisle[3]][j]-coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[3], top_steiner] = 1
                    distance_matrix_tmp[require_in_aisle[3], bottom_steiner] = abs(sum([coords[require_in_aisle[3]][j]-coords[bottom_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[3], bottom_steiner] = 1

                    # 对于最上面的Steiner点
                    distance_matrix_tmp[top_steiner, require_in_aisle[2]] = abs(sum([coords[top_steiner][j]-coords[require_in_aisle[2]][j] for j in range(2)]))
                    neighbor_tmp[top_steiner, require_in_aisle[2]] = 1

                elif len(require_in_aisle) == 3:
                    if (abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[2]][j] for j in range(2)])) >
                            abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[0]][j] for j in range(0)]))):  # gap在上面的情况
                        # 对于最上面的Steiner点
                        distance_matrix_tmp[top_steiner, require_in_aisle[2]] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[2]][j] for j in range(2)]))
                        neighbor_tmp[top_steiner, require_in_aisle[2]] = 1

                        # 对于最上面的必访点（上回撤点）
                        distance_matrix_tmp[require_in_aisle[2], top_steiner] = abs(sum([coords[require_in_aisle[2]][j] - coords[top_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[2], top_steiner] = 1
                        distance_matrix_tmp[require_in_aisle[2], require_in_aisle[0]] = abs(sum([coords[require_in_aisle[2]][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[2], require_in_aisle[0]] = 1
                        distance_matrix_tmp[require_in_aisle[2], bottom_steiner] = abs(sum([coords[require_in_aisle[2]][j] - coords[bottom_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[2], bottom_steiner] = 1

                        # 对于下回撤点
                        distance_matrix_tmp[require_in_aisle[1], require_in_aisle[2]] = abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[2]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], require_in_aisle[2]] = 1
                        distance_matrix_tmp[require_in_aisle[1], bottom_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[bottom_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], bottom_steiner] = 1
                        distance_matrix_tmp[require_in_aisle[1], top_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[top_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], top_steiner] = 1

                        # 对于最下面的必访点
                        distance_matrix_tmp[require_in_aisle[0], bottom_steiner] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        distance_matrix_tmp[require_in_aisle[0], top_steiner] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[0], top_steiner] = 1
                        neighbor_tmp[require_in_aisle[0], bottom_steiner] = 1

                        # 对于最下面的Steiner点
                        distance_matrix_tmp[bottom_steiner, require_in_aisle[1]] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[1]][j] for j in range(2)]))
                        neighbor_tmp[bottom_steiner, require_in_aisle[1]] = 1

                    else:  # gap在下面
                        # 对于最上面的Steiner点
                        distance_matrix_tmp[top_steiner, require_in_aisle[1]] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[1]][j] for j in range(2)]))
                        neighbor_tmp[top_steiner, require_in_aisle[1]] = 1

                        # 对于最上面的必访点
                        distance_matrix_tmp[require_in_aisle[2], top_steiner] = abs(sum([coords[require_in_aisle[2]][j] - coords[top_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[2], top_steiner] = 1
                        distance_matrix_tmp[require_in_aisle[2], bottom_steiner] = abs(sum([coords[require_in_aisle[2]][j] - coords[bottom_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[2], bottom_steiner] = 1

                        # 对于上撤回点
                        distance_matrix_tmp[require_in_aisle[1], require_in_aisle[0]] = abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], require_in_aisle[0]] = 1
                        distance_matrix_tmp[require_in_aisle[1], top_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[top_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], top_steiner] = 1
                        distance_matrix_tmp[require_in_aisle[1], bottom_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[bottom_steiner][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[1], bottom_steiner] = 1

                        # 对于最下面的必访点（下撤回点）
                        distance_matrix_tmp[require_in_aisle[0], bottom_steiner] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        distance_matrix_tmp[require_in_aisle[0], require_in_aisle[2]] = abs(sum([coords[require_in_aisle[2]][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[0], require_in_aisle[2]] = 1
                        neighbor_tmp[require_in_aisle[0], bottom_steiner] = 1
                        distance_matrix_tmp[require_in_aisle[0], top_steiner] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[require_in_aisle[0], top_steiner] = 1

                        # 对于最下面的Steiner点
                        distance_matrix_tmp[bottom_steiner, require_in_aisle[0]] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                        neighbor_tmp[bottom_steiner, require_in_aisle[0]] = 1

                elif len(require_in_aisle)==2:
                    # 对于最上面的Steiner点
                    distance_matrix_tmp[top_steiner, require_in_aisle[1]] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[1]][j] for j in range(2)]))
                    neighbor_tmp[top_steiner, require_in_aisle[1]] = 1

                    # 对于上面的必访点
                    distance_matrix_tmp[require_in_aisle[1], require_in_aisle[0]] = abs(sum([coords[require_in_aisle[1]][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], require_in_aisle[0]] = 1
                    distance_matrix_tmp[require_in_aisle[1], top_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], top_steiner] = 1
                    distance_matrix_tmp[require_in_aisle[1], bottom_steiner] = abs(sum([coords[require_in_aisle[1]][j] - coords[bottom_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[1], bottom_steiner] = 1

                    # 对于下面的必访点
                    distance_matrix_tmp[require_in_aisle[0], bottom_steiner] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    distance_matrix_tmp[require_in_aisle[0], require_in_aisle[1]] = abs(sum([coords[require_in_aisle[0]][j] - coords[require_in_aisle[1]][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[0], require_in_aisle[1]] = 1
                    neighbor_tmp[require_in_aisle[0], bottom_steiner] = 1
                    distance_matrix_tmp[require_in_aisle[0], top_steiner] = abs(sum([coords[require_in_aisle[0]][j] - coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[0], top_steiner] = 1

                    # 对于最下面的Steiner点
                    distance_matrix_tmp[bottom_steiner, require_in_aisle[0]] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[bottom_steiner, require_in_aisle[0]] = 1

                else:  # 只有一个必访点
                    # 对于最上面的Steiner点
                    distance_matrix_tmp[top_steiner, require_in_aisle[0]] = abs(sum([coords[top_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[top_steiner, require_in_aisle[0]] = 1

                    # 对于必访点
                    distance_matrix_tmp[require_in_aisle[0], bottom_steiner] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    distance_matrix_tmp[require_in_aisle[0], top_steiner] = abs(sum([coords[require_in_aisle[0]][j] - coords[top_steiner][j] for j in range(2)]))
                    neighbor_tmp[require_in_aisle[0], top_steiner] = 1
                    neighbor_tmp[require_in_aisle[0], bottom_steiner] = 1

                    # 对于最下面的Steiner点
                    distance_matrix_tmp[bottom_steiner, require_in_aisle[0]] = abs(sum([coords[bottom_steiner][j] - coords[require_in_aisle[0]][j] for j in range(2)]))
                    neighbor_tmp[bottom_steiner, require_in_aisle[0]] = 1

                neighbor_tmp[bottom_steiner,top_steiner]=-1  # 两个Steiner点之间的邻接关系
                neighbor_tmp[top_steiner,bottom_steiner]=-1

            else:  # 如果没有必须访问点在巷道内
                neighbor_tmp[bottom_steiner, top_steiner] = 1  # 两个Steiner点之间的邻接关系
                neighbor_tmp[top_steiner, bottom_steiner] = 1

        # 存横着的Steiner点
        for i in range(0, cross_num):
            for j in range(0, aisle_num - 1):
                distance_matrix_tmp[j + i * (aisle_num), j + i * (aisle_num) + 1] = cross_length
                distance_matrix_tmp[j + i * (aisle_num) + 1, j + i * (aisle_num)] = cross_length
                neighbor_tmp[j + i * (aisle_num), j + i * (aisle_num) + 1] = 1
                neighbor_tmp[j + i * (aisle_num) + 1, j + i * (aisle_num)] = 1

        distance_matrix.append(distance_matrix_tmp)
        neighbor.append(neighbor_tmp)
    return distance_matrix, neighbor

def get_info_aisle_vertex(coords_all_after_delete, require_len_for_batch, steiner_len_for_batch, aisle_num, cross_num):
    '''
    Args:
        coords_all_after_delete: 所有点的坐标 包括depot
        require_len_for_batch:
        steiner_len_for_batch:
        aisle_num:
        cross_num:
        aisle_length:

    Returns: 所有batch每个巷道内点idx，所有batch每个点存在哪个巷道

    '''

    aisle_vertex_info = []  # 存每个巷道内有哪些点 batch,巷道编号,点的idx [[[7,3,4,0][2,5,8,9]]]
    where_is_vertex = []  # 存每个点在哪个巷道里面 batch,点的idx,存在哪个巷道 [[0,1,2,1],[2,1,1]]

    require_len = require_len_for_batch

    for i in range(coords_all_after_delete.size(0)):
        aisle_vertex_info.append([])
        vertex_poition_batch = []
        required_len_for_instancei = require_len[i]  # 这个instance的必访点数量
        steiner_len_for_steineri = steiner_len_for_batch[i]  # 这个instance Steiner点数量 为了不对后面padding的内容进行距离矩阵和邻居的计算

        for _ in range(required_len_for_instancei+steiner_len_for_steineri):
            vertex_poition_batch.append(-1)

        coords = coords_all_after_delete[i].tolist()[0:required_len_for_instancei+steiner_len_for_steineri]

        points_dict_x = {}
        for point_index, (x, y) in enumerate(coords):
            x_val = x  # x的取值
            if x_val in points_dict_x:
                points_dict_x[x_val].append((point_index, y))  # [x]: 点的序号, y
            else:
                points_dict_x[x_val] = [(point_index, y)]  # points_dict_x[x坐标] = [(点的idx, y坐标)]
        # 对字典中每个键对应的值按照元组的第二个元素升序排列
        sorted_dict = {key: sorted(value, key=lambda x: x[1]) for key, value in points_dict_x.items()}  # x取值相同，y升序排列
        # 将排序后每个键对应值的元组的第一个元素取出来，以列表形式存储
        first_elements = {key: [item[0] for item in value] for key, value in sorted_dict.items()}  # 对应的点的idx
        for j in range(0, aisle_num * (cross_num - 1)):  # 遍历所有的巷道
            bottom_steiner = j  # 巷道下面的steiner点
            top_steiner = j + aisle_num  # 巷道上面的Steiner点

            x_axis = coords[bottom_steiner][0]  # 获取下面steiner点的x坐标
            vertex_in_aisle = first_elements[x_axis]  # 把巷道内所有的点取出来
            bottom_steiner_idx = vertex_in_aisle.index(bottom_steiner)  # 下面的Steiner点的idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            top_steiner_idx = vertex_in_aisle.index(top_steiner)  # 上面的Steiner点的Idx 不是在全部坐标里的idx 而是在当前aisle的Idx
            require_in_aisle = vertex_in_aisle[bottom_steiner_idx+1:top_steiner_idx]  # 巷道内所有必须访问点的idx
            steiner_require_in_aisle = vertex_in_aisle[bottom_steiner_idx:top_steiner_idx+1]

            aisle_vertex_info[i].append(steiner_require_in_aisle)
            for v in require_in_aisle:
                vertex_poition_batch[v] = j  # 没必要知道Steiner点在哪
        where_is_vertex.append(vertex_poition_batch)
    return aisle_vertex_info, where_is_vertex