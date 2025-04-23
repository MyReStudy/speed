import math

import numpy as np
import torch


def ACO_solver(required_coords, steiner_coords, steiner_len, where_is_vertex, aisle_vertex_info):
    '''
    Args:
        coords:必须访问点的坐标
        where_is_vertex:
        aisle_vertex_info:
    Returns:
    '''

    def get_dist(idx1, idx2, required_coords, steiner_coords, where_is_vertex, aisle_vertex_info):
        coord1 = required_coords[idx1]
        coord2 = required_coords[idx2]
        steiner_len = len(steiner_coords)
        if coord1[0] == coord2[0]:  # 如果横坐标一样
            return abs(coord1[1] - coord2[1])  # 距离等于纵坐标相减
        aisle1 = where_is_vertex[idx1+steiner_len]  # 得到第一个点所在的巷道
        aisle2 = where_is_vertex[idx2+steiner_len]  # 得到第二个点所在的巷道
        s11 = aisle_vertex_info[aisle1][0]  # 巷道1的上Steiner点
        s12 = aisle_vertex_info[aisle1][-1]  # 巷道1的下Steiner点
        s21 = aisle_vertex_info[aisle2][0]
        s22 = aisle_vertex_info[aisle2][-1]
        coord_s11 = steiner_coords[s11]  # 巷道1上坐标
        coord_s12 = steiner_coords[s12]
        coord_s21 = steiner_coords[s21]
        coord_s22 = steiner_coords[s22]
        d1 = ((abs(coord1[0] - coord_s11[0]) + abs(coord1[1] - coord_s11[1]) +
               abs(coord_s21[0] - coord_s11[0]) + abs(coord_s21[1] - coord_s11[1])) +
              abs(coord_s21[0] - coord2[0]) + abs(coord_s21[1] - coord2[1]))
        d2 = ((abs(coord1[0] - coord_s11[0]) + abs(coord1[1] - coord_s11[1]) +
               abs(coord_s22[0] - coord_s11[0]) + abs(coord_s22[1] - coord_s11[1])) +
              abs(coord_s22[0] - coord2[0]) + abs(coord_s22[1] - coord2[1]))
        d3 = ((abs(coord1[0] - coord_s12[0]) + abs(coord1[1] - coord_s12[1]) +
               abs(coord_s22[0] - coord_s12[0]) + abs(coord_s22[1] - coord_s12[1])) +
              abs(coord_s22[0] - coord2[0]) + abs(coord_s22[1] - coord2[1]))
        d4 = ((abs(coord1[0] - coord_s12[0]) + abs(coord1[1] - coord_s12[1]) +
               abs(coord_s21[0] - coord_s12[0]) + abs(coord_s21[1] - coord_s12[1])) +
              abs(coord_s21[0] - coord2[0]) + abs(coord_s21[1] - coord2[1]))
        return min(d1, d2, d3, d4)

    # 生成距离矩阵
    city_count = len(required_coords)

    dist_matrix = np.zeros((city_count, city_count))

    for i in range(city_count):
        for j in range(i + 1, city_count):
            dist_matrix[i, j] = get_dist(i, j, required_coords, steiner_coords, where_is_vertex, aisle_vertex_info)
            dist_matrix[j, i] = dist_matrix[i, j]  # 距离矩阵对称
        dist_matrix[i,i] = float('inf')

    # print(dist_matrix)

    '''
    文献
    r=0.9, alpha=1, beta=5, tao=0.1, Q=2, m=5
    '''

    # 初始化参数
    # 蚂蚁数量
    AntCount = 5
    # 信息素
    alpha = 1  # 信息素重要程度因子
    beta = 5  # 启发函数重要程度因子
    rho = 0.9  # 挥发速度
    iter = 0  # 迭代初始值
    MAX_iter = 20000  # 最大迭代值
    Q = 2
    tao = 0.1
    # 初始信息素矩阵，全是为1组成的矩阵
    pheromonetable = np.full((city_count, city_count), tao)
    # 候选集列表,存放所有蚂蚁的路径(一只蚂蚁一个路径)
    candidate = np.zeros((AntCount, city_count)).astype(int)
    # path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
    path_best = np.zeros((MAX_iter, city_count))
    # 存放每次迭代的最优距离
    distance_best = np.zeros(MAX_iter)
    # 倒数矩阵
    etable = 1.0 / dist_matrix
    no_change_count=0
    previous_best_distance = float('inf')

    while iter < MAX_iter and no_change_count < 10:
        # first：蚂蚁初始点选择
        if AntCount <= city_count:
            # np.random.permutation随机排列一个数组的
            candidate[:, 0] = np.random.permutation(range(city_count))[:AntCount]
        else:
            m = AntCount - city_count
            n = 2
            candidate[:city_count, 0] = np.random.permutation(range(city_count))[:]
            while m > city_count:
                candidate[city_count * (n - 1):city_count * n, 0] = np.random.permutation(range(city_count))[:]
                m = m - city_count
                n = n + 1
            candidate[city_count * (n - 1):AntCount, 0] = np.random.permutation(range(city_count))[:m]
        length = np.zeros(AntCount)  # 每次迭代的N个蚂蚁的距离值

        # second：选择下一个城市选择
        for i in range(AntCount):
            # 移除已经访问的第一个元素
            unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
            visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
            unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
            for j in range(1, city_count):  # 下一个访问城市的存储位置
                protrans = np.zeros(len(unvisit))  # 转移概率
                for k in range(len(unvisit)):
                    protrans[k] = np.power(pheromonetable[visit][unvisit[k]], alpha) * np.power(
                        etable[visit][unvisit[k]], beta)
                # 累计概率，轮盘赌选择
                cumsumprobtrans = (protrans / sum(protrans)).cumsum()
                cumsumprobtrans -= np.random.rand()
                # 求出离随机数产生最近的索引值
                k = unvisit[list(cumsumprobtrans > 0).index(True)]
                # 下一个访问城市的索引值
                candidate[i, j] = k
                unvisit.remove(k)
                length[i] += dist_matrix[visit][k]
                visit = k
            length[i] += dist_matrix[visit][candidate[i, 0]]

        """
        更新路径等参数
        """
        # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
        if iter == 0:
            distance_best[iter] = length.min()
            path_best[iter] = candidate[length.argmin()].copy()
        else:
            # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
            if length.min() > distance_best[iter - 1]:
                distance_best[iter] = distance_best[iter - 1]
                path_best[iter] = path_best[iter - 1].copy()
            else:  # 当前解比之前的要好，替换当前解和路径
                distance_best[iter] = length.min()
                path_best[iter] = candidate[length.argmin()].copy()

        # 检查是否满足终止条件
        if distance_best[iter] == previous_best_distance:
            no_change_count += 1
        else:
            no_change_count = 0
        previous_best_distance = distance_best[iter]

        """
            信息素的更新
        """
        #信息素的增加量矩阵
        changepheromonetable = np.zeros((city_count, city_count))
        for i in range(AntCount):
            for j in range(city_count - 1):
                # 当前路径比如城市23之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
                changepheromonetable[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
                #Distance[candidate[i, j]][candidate[i, j + 1]]
            #最后一个城市和第一个城市的信息素增加量
            changepheromonetable[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
        #信息素更新的公式：
        pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
        iter += 1
    distance_best = [x for x in distance_best if x != 0]
    return distance_best[-1]