import statsmodels.api as sm
from pre_data import load_and_preprocess_data
import matplotlib.pyplot as plt

# 定义文件路径
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'

# 加载和预处理数据
independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path)

# 对自变量和因变量做最小二乘回归
# 添加常数项到自变量中
X = sm.add_constant(independent_vars)
y1 = dependent_vars[:, 0]  # 第7列为因变量
y2 = dependent_vars[:, 1]  # 第8列为因变量

# 对第7列进行回归分析
model_y1 = sm.OLS(y1, X).fit()
results_y1 = model_y1.summary()
y1_pred = model_y1.predict(X)

# 对第8列进行回归分析
model_y2 = sm.OLS(y2, X).fit()
results_y2 = model_y2.summary()
y2_pred = model_y2.predict(X)

# 打印回归结果
print("回归模型1（第7列作为因变量）结果：")
print(results_y1)

print("\n回归模型2（第8列作为因变量）结果：")
print(results_y2)


# 可视化结果
plt.figure(figsize=(14, 10))

# 第7列回归分析结果
plt.subplot(2, 2, 1)
plt.scatter(y1, y1_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([min(y1), max(y1)], [min(y1), max(y1)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Model 1: Predicted vs Actual (Dependent Variable 1)')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(y1_pred, y1 - y1_pred, alpha=0.6, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Model 1: Residuals vs Predicted')

# 第8列回归分析结果
plt.subplot(2, 2, 3)
plt.scatter(y2, y2_pred, alpha=0.6, color='green', label='Predicted vs Actual')
plt.plot([min(y2), max(y2)], [min(y2), max(y2)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Model 2: Predicted vs Actual (Dependent Variable 2)')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(y2_pred, y2 - y2_pred, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Model 2: Residuals vs Predicted')

plt.tight_layout()
plt.show()