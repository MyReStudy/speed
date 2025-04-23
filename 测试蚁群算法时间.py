import time
import random
import numpy as np
import pandas as pd

from nets.attention_model_stsp import set_neighbor
from problems.stsp.graph_info import get_distance_and_neighbor, get_info_aisle_vertex
from problems.stsp.problem_stsp import generate_instance
from 蚁群算法 import ACO_solver

aco_l = []
seed = []
time_a = []
res = []
df = pd.DataFrame(columns=['aisle_num','corss_num','avg_time','res'])
  # 每次改变仓库规模需要改变三个地方
for aisle_num in range(2,11):
    for cross_num in range(2,11):
        print(f'现在运行到aisle{aisle_num},cross{cross_num}')
        time_e = []
        for i in range(0, 1000):
            random.seed(i)
            seed.append(i)
            aisle_length, cross_length = 20,5
            info = generate_instance(aisle_num, cross_num, aisle_length, cross_length)
            info['depot']=info['depot'].unsqueeze(0)
            info['coords_before_delete'] = info['coords_before_delete'].unsqueeze(0)
            info['deterministic_prize'] = info['deterministic_prize'].unsqueeze(0)
            deterministic_prize = info['deterministic_prize']
            all_loc = info['coords_before_delete']

            steiner_len = info['steiner_len']
            required_len = all_loc.size(1) - steiner_len
            idx_vertex_info = {}
            for i in range(steiner_len):
                idx_vertex_info[i] = 's'
            for i in range(steiner_len, required_len+steiner_len):
                idx_vertex_info[i] = 'v'

            _,neighbor_all = get_distance_and_neighbor(all_loc, [required_len], [steiner_len], aisle_num, cross_num, aisle_length, cross_length)
            set_neighbor(neighbor_all)
            coords = info['coords_before_delete'].tolist()

            aisle_vertex_info, where_is_vertex = get_info_aisle_vertex(all_loc, [required_len], [steiner_len], aisle_num, cross_num)

            required_coords = coords[0][steiner_len:]
            steiner_coords = coords[0][:steiner_len]
            time1 = time.time()
            aco_length = ACO_solver(required_coords, steiner_coords, steiner_len, where_is_vertex[0], aisle_vertex_info[0])
            time2 = time.time()
            time_a.append(time2 - time1)
            res.append(aco_length)

        avg_time = np.mean(time_a)
        avg_res = np.mean(res)

        df = df._append(pd.DataFrame([[aisle_num, cross_num, avg_time, avg_res]], columns=df.columns))


        # 保存到Excel文件
df.to_excel('output-yq.xlsx', index=False)