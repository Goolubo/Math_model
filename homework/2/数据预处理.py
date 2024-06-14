import numpy as np
from sklearn.decomposition import PCA
from pre_data import load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib

# 设置中文
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 加载并预处理数据
independent_vars, dependent_vars, variable_names, _ = load_and_preprocess_data('D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx')

# 最小-最大归一化
def normalize_data(data):

    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()

    # 对自变量进行归一化
    normalized_data = scaler.fit_transform(data)
    return normalized_data


# 仅对自变量进行归一化
normalized_independent_vars = normalize_data(independent_vars)


# 低方差滤波
def low_variance_filter(data, variable_names, threshold=0.05):
    variances = np.var(data, axis=0)
    col_indices = [i for i, var in enumerate(variances) if var > threshold]
    filtered_data = data[:, col_indices]
    filtered_var_names = [variable_names[i] for i in col_indices]

    return filtered_data, filtered_var_names


# 信息熵滤波
def entropy_filter(data, variable_names, threshold=0.3):
    entropies = [entropy(data[:, i]) for i in range(data.shape[1])]
    col_indices = [i for i, entropy in enumerate(entropies) if entropy > threshold]
    filtered_data = data[:, col_indices]
    filtered_var_names = [variable_names[i] for i in col_indices]

    return filtered_data, filtered_var_names


# 计算信息熵
def entropy(data):
    """
    计算数据的信息熵
    :param data: 需要计算信息熵的数据
    :return: 信息熵值
    """
    unique_values = np.unique(data)
    probabilities = [np.count_nonzero(data == value) / len(data) for value in unique_values]
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p != 0])

    return entropy


# 对归一化后的自变量进行低方差滤波和信息熵滤波
independent_var_names = variable_names[14:]  # 提取自变量名称
filtered_independent_vars, filtered_var_names = low_variance_filter(normalized_independent_vars, independent_var_names, threshold=0.05)
filtered_independent_vars, filtered_var_names = entropy_filter(filtered_independent_vars, filtered_var_names, threshold=0.3)

# # 输出过滤后的变量名称和变量个数
# print("过滤后的变量名称:")
# print(filtered_var_names)
# print(f"过滤后的变量个数: {len(filtered_var_names)}")

# PCA降维
pca = PCA(n_components=0.95)  # 保留95%的方差
pca_independent_vars = pca.fit_transform(filtered_independent_vars)

# 获取PCA后的变量名称
pca_var_names = [f'PC{i+1}' for i in range(pca_independent_vars.shape[1])]

# 输出PCA后的变量名称和数量
print("PCA后的变量名称:")
print(pca_var_names)
print(f"PCA后的变量个数: {len(pca_var_names)}")


# 计算累计解释方差比
cum_exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cum_exp_var_ratio)+1), cum_exp_var_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.show()


# 绘制前两个主成分的散点图
plt.figure(figsize=(8, 6))
plt.scatter(pca_independent_vars[:, 0], pca_independent_vars[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Score Plot')
plt.show()

# # 绘制双因子图
# plt.figure(figsize=(10, 8))
# plt.scatter(pca_independent_vars[:, 0], pca_independent_vars[:, 1], alpha=0.5)
# for i, var in enumerate(filtered_var_names):  # 使用过滤后的变量名
#     plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.1, head_length=0.1, color='r')
#     plt.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, var, color='r', ha='center', va='center')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA Biplot')
# plt.show()
