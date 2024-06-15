import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore
from pre_data import load_and_preprocess_data
from pre_data2 import main

# 加载数据
path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
independent_vars, dependent_vars, variable_names, independent_vars_origin = load_and_preprocess_data(path)
pca_independent_vars, pca_var_names = main()

# 合并因变量和PCA降维后的融合变量
combined_vars = np.hstack((independent_vars_origin, pca_independent_vars))

# 对第一个目标变量进行z-score过滤异常值
z_scores = zscore(dependent_vars[:, 0])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 1)  # 保留z-score在3以内的数据

filtered_combined_vars = combined_vars[filtered_entries]
filtered_dependent_vars = dependent_vars[filtered_entries]
filtered_log_dependent_var = np.log(filtered_dependent_vars[:, 0])

# 创建随机森林回归模型实例
reg1 = RandomForestRegressor(n_estimators=100, random_state=42)
reg2 = RandomForestRegressor(n_estimators=100, random_state=42)

# 对过滤后的融合变量和对数变换后的第一个目标变量进行随机森林回归
reg1.fit(filtered_combined_vars, filtered_log_dependent_var)

# 对过滤后的融合变量和第二个目标变量进行随机森林回归
reg2.fit(filtered_combined_vars, filtered_dependent_vars[:, 1])

# 做预测
log_y_pred1 = reg1.predict(filtered_combined_vars)
y_pred2 = reg2.predict(filtered_combined_vars)

# 将第一个目标变量的预测值反变换
y_pred1 = np.exp(log_y_pred1)

# 计算评估指标
mse1 = mean_squared_error(filtered_dependent_vars[:, 0], y_pred1)
r2_1 = r2_score(filtered_dependent_vars[:, 0], y_pred1)

mse2 = mean_squared_error(filtered_dependent_vars[:, 1], y_pred2)
r2_2 = r2_score(filtered_dependent_vars[:, 1], y_pred2)

# 输出评估指标
print(f'目标变量 1 的评估结果:')
print(f'MSE: {mse1:.4f}')
print(f'R²: {r2_1:.4f}')

print(f'目标变量 2 的评估结果:')
print(f'MSE: {mse2:.4f}')
print(f'R²: {r2_2:.4f}')

# 绘制预测图
fig, axs = plt.subplots(1, 2, figsize=(16, 4))

axs[0].scatter(range(len(filtered_dependent_vars)), filtered_dependent_vars[:, 0], label='真实值')
axs[0].plot(range(len(filtered_dependent_vars)), y_pred1, 'r', label='预测值')
axs[0].set_title('目标变量 1')
axs[0].legend()

axs[1].scatter(range(len(filtered_dependent_vars)), filtered_dependent_vars[:, 1], label='真实值')
axs[1].plot(range(len(filtered_dependent_vars)), y_pred2, 'r', label='预测值')
axs[1].set_title('目标变量 2')
axs[1].legend()

plt.tight_layout()
plt.show()
