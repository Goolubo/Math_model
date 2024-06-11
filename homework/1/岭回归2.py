from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 导入预处理模块
from pre_data import load_and_preprocess_data

# 定义文件路径
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'

# 加载和预处理数据
independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(independent_vars)
y_scaled = scaler_y.fit_transform(dependent_vars)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# 使用GridSearchCV进行参数调优
ridge_cv = Ridge()
grid_search = GridSearchCV(ridge_cv, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y_scaled)

# 获取最优参数
best_alpha = grid_search.best_params_['alpha']
print(f'Best alpha: {best_alpha}')

# 使用最优参数训练模型
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)

# 预测
y_train_pred_best = ridge_best.predict(X_train)
y_test_pred_best = ridge_best.predict(X_test)

# 反标准化预测结果
y_train_pred_original_best = scaler_y.inverse_transform(y_train_pred_best)
y_test_pred_original_best = scaler_y.inverse_transform(y_test_pred_best)
y_train_original = scaler_y.inverse_transform(y_train)
y_test_original = scaler_y.inverse_transform(y_test)

# 评估模型
mse_train_best = mean_squared_error(y_train_original, y_train_pred_original_best)
r2_train_best = r2_score(y_train_original, y_train_pred_original_best)
mse_test_best = mean_squared_error(y_test_original, y_test_pred_original_best)
r2_test_best = r2_score(y_test_original, y_test_pred_original_best)

print(f'Training MSE: {mse_train_best:.4f}')
print(f'Training R²: {r2_train_best:.4f}')
print(f'Test MSE: {mse_test_best:.4f}')
print(f'Test R²: {r2_test_best:.4f}')
