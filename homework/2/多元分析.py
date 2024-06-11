import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pre_data import load_and_preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, attention_dim))
        self.linear = nn.Linear(attention_dim, attention_dim)

    def forward(self, x):
        # 计算注意力权重
        weights = torch.softmax(self.attention_weights, dim=0)
        # 加权求和
        out = torch.matmul(x, weights)
        # 映射到 attention_dim
        out = self.linear(out)
        return out


class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim, attention_dim):
        super(MLPWithAttention, self).__init__()
        self.attention_layer = AttentionLayer(input_dim, attention_dim)
        self.fc1 = nn.Linear(attention_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # 应用注意力层
        x = self.attention_layer(x)
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(file_path, sheet_name, input_dim, output_dim, attention_dim, epochs=200, lr=0.0001):
    independent_vars, dependent_vars, _ = load_and_preprocess_data(file_path, sheet_name)

    # 数据归一化
    scaler = StandardScaler()
    independent_vars = scaler.fit_transform(independent_vars)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(independent_vars, dependent_vars, test_size=0.2,
                                                        random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f'训练集独立变量形状: {X_train.shape}')
    print(f'训练集依赖变量形状: {y_train.shape}')
    print(f'测试集独立变量形状: {X_test.shape}')
    print(f'测试集依赖变量形状: {y_test.shape}')

    model = MLPWithAttention(input_dim, output_dim, attention_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
            print(f'第 {epoch + 1}/{epochs} 次迭代，训练损失：{loss.item():.4f}，验证损失：{val_loss.item():.4f}')

    return model


# 使用示例
file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
sheet_name = 'Sheet1'
input_dim = 354  # 根据实际数据调整自变量数
output_dim = 2  # 因变量数
attention_dim = 10  # 降维后的维度

model = train_model(file_path, sheet_name, input_dim, output_dim, attention_dim)
