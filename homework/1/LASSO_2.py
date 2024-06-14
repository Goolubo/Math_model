import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pre_data import load_and_preprocess_data

# 定义文件路径
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'

# 加载和预处理数据
independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path)

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(independent_vars)

# 定义一系列 alpha 值
alphas = np.logspace(-4, 1, 100)

# 初始化系数矩阵
coefs = []

# 对每个 alpha 进行 Lasso 回归
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=50000)  # 增加最大迭代次数
    lasso.fit(X_scaled, dependent_vars[:, 0])  # 这里假设对第一个目标变量进行回归
    coefs.append(lasso.coef_)

# 转换为数组
coefs = np.array(coefs)

# 选择15个自变量进行可视化
selected_indices = np.random.choice(range(coefs.shape[1]), 15, replace=False)
selected_var_names = [variable_names[i] for i in selected_indices]

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

# 绘制系数路径图
plt.figure(figsize=(10, 6))
for i in selected_indices:
    plt.plot(alphas, coefs[:, i], label=variable_names[i])

plt.xscale('log')
plt.xlabel('Alpha', fontproperties=font)
plt.ylabel('系数', fontproperties=font)
plt.title('Lasso系数路径图', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.show()
