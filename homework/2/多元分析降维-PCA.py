import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm





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
            matrix[i, j] = row_data[j] if i == 0 else df.iloc[i + 2, j + 2]

    # 分离自变量和因变量
    # 自变量：矩阵中除去前14列
    independent_vars = matrix[:, [i for i in range(cols) if i not in range(14)]]

    # 因变量：矩阵中的第7列和第8列
    dependent_vars = matrix[:, [6, 7]]

    return independent_vars, dependent_vars, variable_names[14:]


def apply_pca(independent_vars, variable_names, n_components=2):
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(independent_vars)

    # 应用PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # 获取重要特征的索引
    important_indices = np.argsort(-np.abs(pca.components_), axis=1)[:, :5].flatten()
    important_indices = np.unique(important_indices)
    important_features = [variable_names[i] for i in important_indices]

    # 打印重要特征的名称
    print("Important Features:")
    for i, feature in enumerate(important_features):
        print(f"Feature {i + 1}: {feature}")

    return principal_components, pca.explained_variance_ratio_, important_indices, important_features


def plot_correlation_heatmap(independent_vars, variable_names, important_indices):
    # 仅使用重要特征计算相关系数
    df = pd.DataFrame(independent_vars[:, important_indices], columns=[variable_names[i] for i in important_indices])

    # 计算相关系数矩阵
    corr_matrix = df.corr()
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

    # 绘制热图
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=df.columns, yticklabels=df.columns)
    plt.title('Correlation Heatmap of Important Features')
    plt.show()


def main():

    file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'  # 替换为你的文件路径
    sheet_name = 'Sheet1'  # 替换为你的工作表名称

    independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path, sheet_name)

    principal_components, explained_variance_ratio, important_indices, important_features = apply_pca(independent_vars,
                                                                                                      variable_names)

    print("Explained Variance Ratio:")
    print(explained_variance_ratio)

    plot_correlation_heatmap(independent_vars, variable_names, important_indices)


if __name__ == "__main__":
    main()
