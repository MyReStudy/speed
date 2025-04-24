import time
import random
import numpy as np
import pandas as pd

from data_generation import generate_data
from 蚁群算法 import ACO_solver

aco_l = []
seed = []
time_a = []
res = []
df = pd.DataFrame(columns=['aisle_num','corss_num','avg_time','res'])
  # 每次改变仓库规模需要改变三个地方
for aisle_num in range(2,11):
    for cross_num in range(2,11):
        file = f'{aisle_num}_{cross_num}'
        require_info = np.load(f'files/{file}/required_info.npy', allow_pickle=True)
        steiner_info = np.load(f'files/{file}/steiner_info.npy', allow_pickle=True)
        distance_matrix = np.load(f'files/{file}/distance_matrix.npy', allow_pickle=True)
        print(f'现在运行到aisle{aisle_num},cross{cross_num}')
        time_e = []
        for i in range(0, 50):
            random.seed(i)
            seed.append(i)
            info = generate_data(aisle_num, cross_num, require_info)


            aco_length,time_aco_tmp = ACO_solver(info['orders_idx'], distance_matrix)

            time_a.append(time_aco_tmp)
            res.append(aco_length)

        avg_time = np.mean(time_a)
        avg_res = np.mean(res)

        print(avg_time)

        df = df._append(pd.DataFrame([[aisle_num, cross_num, avg_time, avg_res]], columns=df.columns))


        # 保存到Excel文件
df.to_excel('output-yq.xlsx', index=False)