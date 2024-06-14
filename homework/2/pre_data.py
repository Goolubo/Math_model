import pandas as pd
import numpy as np


def load_and_preprocess_data(file_path, sheet_name='Sheet1'):
    # 加载 Excel 文件
    xls = pd.ExcelFile(file_path)

    # 加载工作表
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 提取变量名
    variable_names = df.iloc[1, 2:].tolist()

    # 并构建矩阵
    row_data = df.iloc[2].values[2:]
    column_data = df.iloc[2:, 2].values

    # 去除任何可能存在的 NaN 值
    row_data = row_data[~pd.isnull(row_data)]
    column_data = column_data[~pd.isnull(column_data)]

    # 确定矩阵的实际大小
    rows = len(column_data)
    cols = len(row_data)
    matrix = np.zeros((rows, cols))

    # 填充矩阵
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = row_data[j] if i == 0 else df.iloc[i+2, j+2]

    # 分离自变量和因变量
    # 需处理的自变量：矩阵中除去前14列
    independent_vars = matrix[:, [i for i in range(cols) if i not in range(14)]]
    # 只有原料性质的自变量
    independent_vars_origin = matrix[:, [i for i in range(cols) if i in range(0, 7)]]

    # 因变量：矩阵中的第6列和第7列
    dependent_vars = matrix[:, [7, 8]]

    return independent_vars, dependent_vars, variable_names, independent_vars_origin

