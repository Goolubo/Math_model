from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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

# 创建线性回归模型
linreg = LinearRegression()

# 训练模型
linreg.fit(X_train, y_train)

# 预测
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# 反标准化预测结果
y_train_pred_original = scaler_y.inverse_transform(y_train_pred)
y_test_pred_original = scaler_y.inverse_transform(y_test_pred)
y_train_original = scaler_y.inverse_transform(y_train)
y_test_original = scaler_y.inverse_transform(y_test)

# 评估模型
for i in range(y_train_original.shape[1]):
    mse_train = mean_squared_error(y_train_original[:, i], y_train_pred_original[:, i])
    r2_train = r2_score(y_train_original[:, i], y_train_pred_original[:, i])
    mse_test = mean_squared_error(y_test_original[:, i], y_test_pred_original[:, i])
    r2_test = r2_score(y_test_original[:, i], y_test_pred_original[:, i])

    print(f'目标变量 {i + 1}:')
    print(f'训练集 MSE: {mse_train:.4f}')
    print(f'训练集 R²: {r2_train:.4f}')
    print(f'测试集 MSE: {mse_test:.4f}')
    print(f'测试集 R²: {r2_test:.4f}')
    print()

# 可视化结果
num_targets = y_train_original.shape[1]  # 因变量数量

plt.figure(figsize=(14, 6 * num_targets))

for i in range(num_targets):
    plt.subplot(num_targets, 2, 2*i + 1)
    plt.scatter(y_train_original[:, i], y_train_pred_original[:, i], alpha=0.6, color='blue', label='预测值 vs 实际值')
    plt.plot([min(y_train_original[:, i]), max(y_train_original[:, i])], [min(y_train_original[:, i]), max(y_train_original[:, i])], color='red', linestyle='--', label='理想情况')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'训练集: 目标变量 {i+1} 预测值 vs 实际值')
    plt.legend()

    plt.subplot(num_targets, 2, 2*i + 2)
    plt.scatter(y_test_original[:, i], y_test_pred_original[:, i], alpha=0.6, color='green', label='预测值 vs 实际值')
    plt.plot([min(y_test_original[:, i]), max(y_test_original[:, i])], [min(y_test_original[:, i]), max(y_test_original[:, i])], color='red', linestyle='--', label='理想情况')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'测试集: 目标变量 {i+1} 预测值 vs 实际值')
    plt.legend()

plt.tight_layout()
plt.show()