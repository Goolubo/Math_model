import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pre_data import load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'


# def print_stats(data, description):
#     """
#     检验是否有正确归一化
#     """
#     print(f"{description} - Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")

def normalize_data(data):

    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()

    # 对自变量进行归一化
    normalized_data = scaler.fit_transform(data)
    # print_stats(normalized_data, "After normalization")
    return normalized_data


def low_variance_filter(data, variable_names, threshold=0.05):
    variances = np.var(data, axis=0)
    col_indices = [i for i, var in enumerate(variances) if var > threshold]
    filtered_data = data[:, col_indices]
    filtered_var_names = [variable_names[i] for i in col_indices]
    # print_stats(filtered_data, "After low variance filter")
    return filtered_data, filtered_var_names


def entropy(data):
    """
    计算数据的信息熵
    :param data: 需要计算信息熵的数据
    :return: 信息熵值
    """
    unique_values = np.unique(data)
    probabilities = [np.count_nonzero(data == value) / len(data) for value in unique_values]
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p != 0])
    return entropy_value


def entropy_filter(data, variable_names, threshold=0.3):
    entropies = [entropy(data[:, i]) for i in range(data.shape[1])]
    col_indices = [i for i, ent in enumerate(entropies) if ent > threshold]
    filtered_data = data[:, col_indices]
    filtered_var_names = [variable_names[i] for i in col_indices]
    # print_stats(filtered_data, "After entropy filter")
    return filtered_data, filtered_var_names


def perform_pca(independent_vars, variable_names, variance_threshold=0.95):
    normalized_independent_vars = normalize_data(independent_vars)
    independent_var_names = variable_names[14:]  # Adjust as needed
    filtered_independent_vars, filtered_var_names = low_variance_filter(normalized_independent_vars,
                                                                        independent_var_names)
    filtered_independent_vars, filtered_var_names = entropy_filter(filtered_independent_vars, filtered_var_names)

    pca = PCA(n_components=variance_threshold)
    pca_independent_vars = pca.fit_transform(filtered_independent_vars)
    # print_stats(pca_independent_vars, "After PCA")
    pca_var_names = [f'PC{i + 1}' for i in range(pca_independent_vars.shape[1])]

    return pca_independent_vars, pca_var_names, pca


def Out_combined_vars(data):
    # 将 combined_vars 转化为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存到 Excel 文件
    excel_filename = 'combined_vars.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f"{excel_filename} has been created successfully.")


def main():
    independent_vars, dependent_vars, variable_names, independent_vars_origin = load_and_preprocess_data(
        'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx')
    pca_independent_vars, pca_var_names, pca = perform_pca(independent_vars, variable_names)

    # print("PCA后的变量名称:")
    # print(pca_var_names)
    # print(f"PCA后的变量个数: {len(pca_var_names)}")

    return pca_independent_vars, pca_var_names


if __name__ == "__main__":
    pca_independent_vars, pca_var_names = main()
