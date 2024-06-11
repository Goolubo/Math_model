import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pre_data import load_and_preprocess_data
from sklearn.linear_model import lasso_path

# 定义文件路径
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'

# 加载和预处理数据
independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path)

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(independent_vars)

# 初始化列表存储结果
best_alphas = []
mse_train_list = []
r2_train_list = []
mse_test_list = []
r2_test_list = []

# 创建列表来保存反标准化后的预测值
y_train_preds_original = []
y_test_preds_original = []
y_train_original_list = []
y_test_original_list = []

# 分别对两个目标变量进行LASSO回归
for i in range(dependent_vars.shape[1]):
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(dependent_vars[:, i].reshape(-1, 1))

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 定义参数网格
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

    # 使用GridSearchCV进行参数调优
    lasso_cv = Lasso(max_iter=50000)
    grid_search = GridSearchCV(lasso_cv, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 获取最优参数
    best_alpha = grid_search.best_params_['alpha']
    best_alphas.append(best_alpha)
    print(f'Best alpha for target {i + 1}: {best_alpha}')

    # 使用最优参数训练模型
    lasso_best = Lasso(alpha=best_alpha, max_iter=50000)  # 增加迭代次数
    lasso_best.fit(X_train, y_train)

    # 预测
    y_train_pred = lasso_best.predict(X_train).reshape(-1, 1)
    y_test_pred = lasso_best.predict(X_test).reshape(-1, 1)

    # 反标准化预测结果
    y_train_pred_original = scaler_y.inverse_transform(y_train_pred)
    y_test_pred_original = scaler_y.inverse_transform(y_test_pred)
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    y_train_preds_original.append(y_train_pred_original)
    y_test_preds_original.append(y_test_pred_original)
    y_train_original_list.append(y_train_original)
    y_test_original_list.append(y_test_original)

    # 评估模型
    mse_train = mean_squared_error(y_train_original, y_train_pred_original)
    r2_train = r2_score(y_train_original, y_train_pred_original)
    mse_test = mean_squared_error(y_test_original, y_test_pred_original)
    r2_test = r2_score(y_test_original, y_test_pred_original)

    mse_train_list.append(mse_train)
    r2_train_list.append(r2_train)
    mse_test_list.append(mse_test)
    r2_test_list.append(r2_test)

    print(f'\nTarget {i + 1}:')
    print(f'Training MSE: {mse_train:.4f}')
    print(f'Training R²: {r2_train:.4f}')
    print(f'Test MSE: {mse_test:.4f}')
    print(f'Test R²: {r2_test:.4f}')

# 将结果转为数组以便绘图
y_train_preds_original = np.hstack(y_train_preds_original)
y_test_preds_original = np.hstack(y_test_preds_original)
y_train_original_list = np.hstack(y_train_original_list)
y_test_original_list = np.hstack(y_test_original_list)

# 可视化结果
plt.figure(figsize=(14, 10))

for i in range(dependent_vars.shape[1]):
    plt.subplot(2, dependent_vars.shape[1], i + 1)
    plt.scatter(y_train_original_list[:, i], y_train_preds_original[:, i], alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_train_original_list[:, i]), max(y_train_original_list[:, i])],
             [min(y_train_original_list[:, i]), max(y_train_original_list[:, i])], color='red', linestyle='--',
             label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Training Set: Target {i + 1}')
    plt.legend()

    plt.subplot(2, dependent_vars.shape[1], dependent_vars.shape[1] + i + 1)
    plt.scatter(y_test_original_list[:, i], y_test_preds_original[:, i], alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test_original_list[:, i]), max(y_test_original_list[:, i])],
             [min(y_test_original_list[:, i]), max(y_test_original_list[:, i])], color='red', linestyle='--',
             label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Test Set: Target {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()


# 设置matplotlib配置，确保中文和负号可以正确显示
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 设置字体列表，增加SimHei作为备选
plt.rcParams['axes.unicode_minus'] = False  # 确保能正常显示负号



# 定义一个更宽广的alpha范围以便观察系数变化
alphas = np.logspace(-4, 2, 200)

# 选择要展示的变量的下标，例如，这里选择了前5个变量
selected_indices = list(range(10))
selected_variable_names = [variable_names[i] for i in selected_indices]

# 计算LASSO路径
_, coefs, _ = lasso_path(X_scaled[:, selected_indices], dependent_vars[:, 0], alphas=alphas)

# 绘制LASSO系数路径图
plt.figure(figsize=(10, 6))
for i in range(len(selected_indices)):
    plt.plot(alphas, coefs[i, :], label=f'Coefficient {selected_variable_names[i]}')

plt.gca().invert_xaxis()  # 因为alpha减小时，正则化效果减弱，通常习惯从右到左阅读图
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Coefficients Path')
plt.legend(loc='upper right')
plt.xscale('log')
plt.grid(True)
plt.show()

# 绘制LASSO正则化路径图，展示每个alpha下的模型性能（MSE）
mse_path = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train[:, selected_indices], y_train)
    mse = mean_squared_error(y_test, lasso.predict(X_test[:, selected_indices]))
    mse_path.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(alphas, mse_path, label='Test MSE')
plt.gca().invert_xaxis()
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('LASSO Regularization Path')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.show()