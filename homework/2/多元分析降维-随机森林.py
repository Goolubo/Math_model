import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from pre_data import load_and_preprocess_data



def feature_importance_rf(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    return importances


def select_important_features(X, importances, threshold=0.01):
    important_indices = np.where(importances > threshold)[0]
    return X[:, important_indices], important_indices


def plot_correlation_matrix(X, feature_names):
    # 创建一个 DataFrame 从已选择的特征
    df = pd.DataFrame(X, columns=feature_names)

    # 计算相关性矩阵
    corr_matrix = df.corr()
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('降维后特征的相关性矩阵')
    plt.show()


def main(file_path, sheet_name='Sheet1'):
    X, y, variable_names, X_origin = load_and_preprocess_data(file_path, sheet_name)
    y = y[:, 0]  # 假设我们使用第一个因变量进行随机森林回归

    # 步骤1：使用随机森林进行特征重要性评分
    importances = feature_importance_rf(X, y)

    # 步骤2：选择重要特征
    X_selected, important_indices = select_important_features(X, importances)

    # 打印重要特征的名称
    important_feature_names = [variable_names[i] for i in important_indices]
    print("重要特征:", important_feature_names)

    # 步骤3：绘制已选择特征的相关性矩阵
    plot_correlation_matrix(X_selected, important_feature_names)

    return X_selected, important_feature_names


# 使用示例
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
sheet_name = 'Sheet1'
X_selected, important_feature_names = main(file_path, sheet_name)
print("降维后特征的形状:", X_selected.shape)
