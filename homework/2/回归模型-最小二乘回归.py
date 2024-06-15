import numpy as np
from pre_data import load_and_preprocess_data
from pre_data2 import main
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
independent_vars, dependent_vars, variable_names, independent_vars_origin = load_and_preprocess_data(path)
pca_independent_vars, pca_var_names = main()

# 合并因变量和PCA降维后的融合变量
combined_vars = np.hstack((independent_vars_origin, pca_independent_vars))

# 假设dependent_vars的形状是(n_samples, n_targets)
n_targets = dependent_vars.shape[1]

fig, axs = plt.subplots(1, n_targets, figsize=(16, 4))

for i in range(n_targets):
    # 创建线性回归模型实例
    reg = LinearRegression()

    # 对融合变量和第i个目标变量进行线性回归
    reg.fit(combined_vars, dependent_vars[:, i])

    # 做预测
    y_pred = reg.predict(combined_vars)

    # 计算评估指标
    mse = mean_squared_error(dependent_vars[:, i], y_pred)
    r2 = r2_score(dependent_vars[:, i], y_pred)

    # 输出评估指标
    print(f'目标变量 {i + 1} 的评估结果:')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')

    # 绘制预测图
    ax = axs[i] if n_targets > 1 else axs
    ax.scatter(range(len(dependent_vars)), dependent_vars[:, i], label='真实值')
    ax.plot(range(len(dependent_vars)), y_pred, 'r', label='预测值')
    ax.set_title(f'目标变量 {i + 1}')
    ax.legend()

plt.tight_layout()
plt.show()