import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression


def load_and_preprocess_data(file_path, sheet_name='Sheet1'):
    # 加载 Excel 文件
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
            matrix[i, j] = df.iloc[i + 2, j + 2]

    # 分离自变量和因变量
    # 自变量：矩阵中除去前14列
    independent_vars = matrix[:, 14:]

    # 因变量：矩阵中的第7列和第8列
    dependent_vars = matrix[:, [6, 7]]

    return independent_vars, dependent_vars, variable_names[14:]


def perform_kernel_pca(independent_vars, kernel='rbf', n_components=2, gamma=None):
    kpca = KernelPCA(kernel=kernel, n_components=n_components, gamma=gamma)
    transformed_data = kpca.fit_transform(independent_vars)
    return transformed_data, kpca


def get_important_features(independent_vars, transformed_data, variable_names, top_n=15):
    # 使用线性回归模型来评估特征重要性
    lin_reg = LinearRegression()
    lin_reg.fit(transformed_data, independent_vars)
    feature_importances = np.mean(np.abs(lin_reg.coef_), axis=0)

    indices = np.argsort(feature_importances)[::-1][:top_n]
    important_variable_names = np.array(variable_names)[indices]
    return important_variable_names


def main(file_path, sheet_name='Sheet1', kernel='rbf', n_components=2, gamma=None, top_n=15):
    independent_vars, dependent_vars, variable_names, independent_vars_origin = load_and_preprocess_data(file_path, sheet_name)

    # Perform Kernel PCA
    transformed_data, kpca = perform_kernel_pca(independent_vars, kernel, n_components, gamma)

    # 获取重要变量名称
    important_variable_names = get_important_features(independent_vars, transformed_data, variable_names, top_n)

    # Output transformed data and important variable names
    return transformed_data, important_variable_names


# 调用 main 函数处理数据并进行降维
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
transformed_data, important_variable_names = main(file_path)

print("Transformed Data:")
print(transformed_data)

print("Important Variable Names:")
print(important_variable_names)
