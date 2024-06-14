import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(file_path, sheet_name='Sheet1'):
    # 加载 Excel 文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 提取变量名
    variable_names = df.iloc[1, 2:].tolist()

    # 并构建矩阵
    row_data = df.iloc[2].values[2:]
    column_data = df.iloc[2:, 2].values

    # 去除任何可能存在的 NaN 值
    row_data = row_data[~pd.isnull(row_data)]
    column_data = column_data[~pd.isnull(column_data)]

    # 确定矩阵的实际大小
    rows = len(column_data)
    cols = len(row_data)
    matrix = np.zeros((rows, cols))

    # 填充矩阵
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = row_data[j] if i == 0 else df.iloc[i + 2, j + 2]

    # 分离自变量和因变量
    # 自变量：矩阵中除去前14列
    independent_vars = matrix[:, [i for i in range(cols) if i not in range(14)]]

    # 因变量：矩阵中的第7列和第8列
    dependent_vars = matrix[:, [6, 7]]

    return independent_vars, dependent_vars, variable_names


file_path = 'D:/桌面/汽油辛烷值模型/附件一：325个样本数据.xlsx'
independent_vars, dependent_vars, variable_names = load_and_preprocess_data(file_path)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


input_dim = independent_vars.shape[1]
hidden_dim = 128
latent_dim = 32

autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# 准备数据
independent_vars_tensor = torch.tensor(independent_vars, dtype=torch.float32)
dataset = TensorDataset(independent_vars_tensor, independent_vars_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练自编码器
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 获取降维后的自变量
with torch.no_grad():
    reduced_vars = autoencoder.encoder(independent_vars_tensor).numpy()

# 输出降维后的自变量名称
reduced_var_names = [f'latent_var_{i + 1}' for i in range(latent_dim)]
print(reduced_var_names)
