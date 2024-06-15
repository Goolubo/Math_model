import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pre_data import load_and_preprocess_data
from pre_data2 import main
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
independent_vars, dependent_vars, variable_names, independent_vars_origin = load_and_preprocess_data(path)
pca_independent_vars, pca_var_names = main()

# 归一化处理
# scaler = MinMaxScaler()
# independent_vars_scaled = scaler.fit_transform(independent_vars_origin)
# pca_independent_scaled = scaler.fit_transform(pca_independent_vars)

# 合并因变量和PCA降维后的融合变量
combined_vars = np.hstack((independent_vars_origin, pca_independent_vars))
# print(combined_vars.shape)
"""
independent_vars_origin:(325, 7)
pca_independent_vars:(325, 11)
combined_vars:(325, 18)
"""
# 检查合并后的数据集
# Out_combined_vars(combined_vars)

# 将Numpy数组转换为PyTorch张量
X = torch.from_numpy(combined_vars).float()
y1 = torch.from_numpy(np.log1p(dependent_vars[:, 0])).float().unsqueeze(1)  # 对第一个因变量进行对数变换
y2 = torch.from_numpy(dependent_vars[:, 1]).float().unsqueeze(1)

# 划分数据集为训练集和测试集
train_ratio = 0.8
train_size = int(X.shape[0] * train_ratio)
train_X, test_X = X[:train_size], X[train_size:]
train_y1, test_y1 = y1[:train_size], y1[train_size:]
train_y2, test_y2 = y2[:train_size], y2[train_size:]

# 创建PyTorch数据集和数据加载器
train_dataset1 = TensorDataset(train_X, train_y1)
train_dataset2 = TensorDataset(train_X, train_y2)
test_dataset1 = TensorDataset(test_X, test_y1)
test_dataset2 = TensorDataset(test_X, test_y2)
train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=32, shuffle=False)

# 定义深度学习模型
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(RegressionModel, self).__init__()
        layers = []
        self.hidden_sizes = hidden_sizes
        for i in range(len(hidden_sizes)):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_sizes[i-1]
            output_dim = hidden_sizes[i]
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# 实例化两个模型
input_size = X.shape[1]
model1 = RegressionModel(input_size)
model2 = RegressionModel(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    running_loss1 = 0.0
    running_loss2 = 0.0
    for inputs, labels1 in train_loader1:
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels1)
        loss.backward()
        optimizer1.step()
        running_loss1 += loss.item()
    for inputs, labels2 in train_loader2:
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels2)
        loss.backward()
        optimizer2.step()
        running_loss2 += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {running_loss1/len(train_loader1):.4f}, Loss2: {running_loss2/len(train_loader2):.4f}')

# 在测试集上评估模型并获取预测值
model1.eval()
model2.eval()
test_preds1 = []
test_labels1 = []
test_preds2 = []
test_labels2 = []
with torch.no_grad():
    for inputs, labels in test_loader1:
        outputs = model1(inputs)
        test_preds1.extend(outputs.numpy())
        test_labels1.extend(labels.numpy())
    for inputs, labels in test_loader2:
        outputs = model2(inputs)
        test_preds2.extend(outputs.numpy())
        test_labels2.extend(labels.numpy())

test_preds1 = np.array(test_preds1)
test_labels1 = np.array(test_labels1)
test_preds2 = np.array(test_preds2)
test_labels2 = np.array(test_labels2)

# 反对数变换第一个因变量的预测值
test_preds1 = np.expm1(test_preds1)
test_labels1 = np.expm1(test_labels1)

# 计算MSE和R^2
mse1 = mean_squared_error(test_labels1, test_preds1)
r2_1 = r2_score(test_labels1, test_preds1)
mse2 = mean_squared_error(test_labels2, test_preds2)
r2_2 = r2_score(test_labels2, test_preds2)

# 打印结果
print(f'目标变量 1 的评估结果:')
print(f'MSE: {mse1:.4f}')
print(f'R²: {r2_1:.4f}')
print(f'目标变量 2 的评估结果:')
print(f'MSE: {mse2:.4f}')
print(f'R²: {r2_2:.4f}')


# 绘制预测图
plt.figure()
plt.scatter(test_labels1, test_preds1, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction for Output 1')
plt.plot([test_labels1.min(), test_labels1.max()], [test_labels1.min(), test_labels1.max()], 'r--', lw=2)
plt.show()

plt.figure()
plt.scatter(test_labels2, test_preds2, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction for Output 2')
plt.plot([test_labels2.min(), test_labels2.max()], [test_labels2.min(), test_labels2.max()], 'r--', lw=2)
plt.show()
