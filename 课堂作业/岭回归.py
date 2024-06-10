import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# 训练岭回归模型
ridge_model = Ridge(alpha=1.0)  # alpha是正则化强度，可以调整以获得更好的效果
ridge_model.fit(X_train, y_train)

# 预测
y_train_pred = ridge_model.predict(X_train)
y_test_pred = ridge_model.predict(X_test)

# 反标准化预测结果
y_train_pred_original = scaler_y.inverse_transform(y_train_pred)
y_test_pred_original = scaler_y.inverse_transform(y_test_pred)
y_train_original = scaler_y.inverse_transform(y_train)
y_test_original = scaler_y.inverse_transform(y_test)

# 评估模型
mse_train = mean_squared_error(y_train_original, y_train_pred_original)
r2_train = r2_score(y_train_original, y_train_pred_original)
mse_test = mean_squared_error(y_test_original, y_test_pred_original)
r2_test = r2_score(y_test_original, y_test_pred_original)

print(f'Training MSE: {mse_train:.4f}')
print(f'Training R²: {r2_train:.4f}')
print(f'Test MSE: {mse_test:.4f}')
print(f'Test R²: {r2_test:.4f}')


# 可视化结果
plt.figure(figsize=(14, 6))

# 训练集预测值与实际值
plt.subplot(1, 2, 1)
plt.scatter(y_train_original[:, 0], y_train_pred_original[:, 0], alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([min(y_train_original[:, 0]), max(y_train_original[:, 0])], [min(y_train_original[:, 0]), max(y_train_original[:, 0])], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set: Predicted vs Actual')
plt.legend()

# 测试集预测值与实际值
plt.subplot(1, 2, 2)
plt.scatter(y_test_original[:, 0], y_test_pred_original[:, 0], alpha=0.6, color='green', label='Predicted vs Actual')
plt.plot([min(y_test_original[:, 0]), max(y_test_original[:, 0])], [min(y_test_original[:, 0]), max(y_test_original[:, 0])], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set: Predicted vs Actual')
plt.legend()

plt.tight_layout()
plt.show()