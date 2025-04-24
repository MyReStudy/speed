import time
import random
import numpy as np
import pandas as pd

# from problems import STSP
from data_generation import generate_data

deep_l1=[]
deep_l2 = []
exact_l = []
seed = []
time_e = []
time_d = []
res = []
df = pd.DataFrame(columns=['aisle_num','corss_num','avg_time','avg_res'])
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

def exact_solver1(aisle_vertex_info,coords, aisle_num, corss_num):
    v_aisle = aisle_num
    c_aisle = corss_num

    neighbors = {}
    for x in aisle_vertex_info:
        for j in x:
            neighbors[j] = []
    for x in aisle_vertex_info:
        for j in range(1, len(x)):
            neighbors[x[j]].append(x[j - 1])
            neighbors[x[j - 1]].append(x[j])
    aisles_num_in_block = v_aisle
    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle].append(aisle + aisles_num_in_block * cross_aisle + 1)

    for cross_aisle in range(0, c_aisle):
        for aisle in range(0, v_aisle - 1):
            neighbors[aisle + aisles_num_in_block * cross_aisle + 1].append(aisle + aisles_num_in_block * cross_aisle)

    edge_dict = {}
    for x in neighbors:
        for y in neighbors[x]:
            edge_dict[(x, y)] = abs(coords[x][0] - coords[y][0]) + abs(coords[x][1] - coords[y][1])

    # required_vertex = [i for i in range(12,36)]
    # for i in range
    # edge_dict = 1
    # neighbors = 2
    steiner_vertex = [i for i in range(0, v_aisle * c_aisle)]
    required_vertex = [i for i in range(v_aisle * c_aisle, len(coords))]
    all_vertex = steiner_vertex + required_vertex
    time1 = time.time()
    mdl = Model('STSP')
    mdl.Params.TimeLimit = 1800
    mdl.setParam('OutputFlag', 0)

    edge = list(edge_dict.keys())  # 所有的边
    N = len(required_vertex)  # 必访点数量

    x = {}
    y = {}

    for e in edge:
        x[e] = mdl.addVar(vtype=GRB.BINARY, name='x_' + str(e[0]) + '_' + str(e[1]))
        y[e] = mdl.addVar(vtype=GRB.CONTINUOUS, name='y_' + str(e[0]) + '_' + str(e[1]))

    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[e] * edge_dict[e] for e in edge))

    mdl.addConstrs(quicksum(x[i, j] for j in neighbors[i]) >= 1 for i in required_vertex)

    for i in all_vertex:
        expr1 = LinExpr(0)
        expr2 = LinExpr(0)
        for j in neighbors[i]:
            expr1.addTerms(1, x[i, j])
            expr2.addTerms(1, x[j, i])
        mdl.addConstr(expr1 == expr2)
        expr1.clear()
        expr2.clear()

    for i in required_vertex[1:]:
        expr3 = LinExpr(0)
        expr4 = LinExpr(0)
        for j in neighbors[i]:
            expr3.addTerms(1, y[i, j])
            expr4.addTerms(1, y[j, i])
        mdl.addConstr(-expr3 - 1 == -expr4)
        expr3.clear()
        expr4.clear()

    for i in steiner_vertex:
        expr5 = LinExpr(0)
        expr6 = LinExpr(0)
        for j in neighbors[i]:
            expr5.addTerms(1, y[i, j])
            expr6.addTerms(1, y[j, i])
        mdl.addConstr(expr5 == expr6)
        expr5.clear()
        expr6.clear()

    for e in edge:
        mdl.addConstr(y[e[0], e[1]] <= N * x[e[0], e[1]])

    mdl.optimize()  # 优化
    obj_res = mdl.getObjective().getValue()
    time2 = time.time()
    # print(mdl.PoolObjBound)
    time_e_tmp = time2-time1
    return obj_res, time_e_tmp, mdl

  # 每次改变仓库规模需要改变三个地方
for aisle_num in range(2,11):
    for cross_num in range(2,11):
        file = f'{aisle_num}_{cross_num}'
        require_info = np.load(f'files/speed/{file}/required_info.npy', allow_pickle=True)
        steiner_info = np.load(f'files/speed/{file}/steiner_info.npy', allow_pickle=True)
        distance_matrix = np.load(f'files/speed/{file}/distance_matrix.npy', allow_pickle=True)
        print(f'现在运行到aisle{aisle_num},cross{cross_num}')
        time_e = []
        for i in range(0, 1000):
            random.seed(i)
            seed.append(i)
            info_stsp = generate_data(require_info)

            required_len = len(info_stsp['all_loc_before_delete'].tolist())
            steiner_len = len(steiner_info[0])

            all_loc = torch.tensor(steiner_info[3].tolist() + info_stsp['all_loc_before_delete'].tolist(),
                                   device='cuda').unsqueeze(0)
            aisle_vertex_info, where_is_vertex = get_info_aisle_vertex(all_loc, [required_len], [steiner_len],
                                                                       aisle_num, cross_num)
            coords = all_loc.squeeze(0).tolist()

            ex_length, time_e_tmp, mdl = exact_solver1(aisle_vertex_info[0],coords, aisle_num, cross_num)
            time_e.append(time_e_tmp)
            res.append(ex_length)
        avg_time = np.mean(time_e)
        avg_res = np.mean(res)

        df = df._append(pd.DataFrame([[aisle_num, cross_num, avg_time,avg_res]], columns=df.columns))

        # 保存到Excel文件
df.to_excel('output-jq.xlsx', index=False)