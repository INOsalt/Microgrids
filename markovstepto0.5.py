import os
import numpy as np
import pandas as pd


def read_and_multiply_matrices(input_dir, output_dir, step=10):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中所有CSV文件的列表并排序
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    # 分批处理每十个文件
    for i in range(0, len(files), step):
        batch_files = files[i:i + step]
        matrices = []

        # 读取并收集当前批次的矩阵
        for filename in batch_files:
            path = os.path.join(input_dir, filename)
            matrix = pd.read_csv(path).values
            matrices.append(matrix)

        # 只有当我们有足够的矩阵时才进行乘法操作
        if len(matrices) == step:
            result_matrix = np.linalg.multi_dot(matrices)
            # 保存结果矩阵到CSV
            result_path = os.path.join(output_dir, f"TMhalfhour_{i // step:02}.csv")
            pd.DataFrame(result_matrix).to_csv(result_path, header=None, index=False)
        else:
            print(f"Skipping batch starting at {i} due to insufficient matrices.")


#示例用法
input_dir = 'TM'  # 这是包含每三分钟转移矩阵CSV文件的目录路径
output_dir = 'TMhalfhour'  # 这是结果将被保存的目录路径
read_and_multiply_matrices(input_dir, output_dir)

import os
import numpy as np
import pandas as pd


def calculate_state_vectors(initial_state, tm_folder, output_file):
    # 确保TMhalfhour文件夹存在
    if not os.path.exists(tm_folder):
        print(f"Folder '{tm_folder}' does not exist.")
        return

    # 读取TMhalfhour文件夹中所有转移矩阵文件的列表并排序
    tm_files = sorted([f for f in os.listdir(tm_folder) if f.endswith('.csv')])

    # 初始化状态向量为初始状态
    current_state = np.array(initial_state)

    # 用于存储每个时刻的状态向量
    state_vectors = [current_state]

    # 遍历并应用每个转移矩阵
    for tm_file in tm_files:
        tm_path = os.path.join(tm_folder, tm_file)
        tm = pd.read_csv(tm_path, header=None).values
        current_state = np.dot(current_state, tm)
        state_vectors.append(current_state)

    # 保存每半个小时的状态向量到CSV文件
    pd.DataFrame(state_vectors).to_csv(output_file, header=None, index=False)
    print(f"Saved state vectors to '{output_file}'.")


# 示例用法
initial_state = [0., 0., 0., 0., 0., 0., 0., 1000., 1000., 1000.,
                 1000., 1000., 0., 1000., 1000., 0., 1000., 1000., 1000., 1000.,
                 1000., 1000., 1000., 1000., 1000., 0., 0., 1000., 1000., 1000.,
                 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0.]

tm_folder = 'TMhalfhour'
output_file = 'SVhalfhour.csv'

calculate_state_vectors(initial_state, tm_folder, output_file)

