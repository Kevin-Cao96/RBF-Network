import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


# ====== 定义 CNN ======
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：1个通道 → 16个通道，3×3卷积核
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)       # 2×2 池化，长宽减半
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 经过两次卷积+池化后，8×8 → 4×4 → 2×2，32个通道
        # 所以全连接层输入是 2×2×32 = 128
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 8×8 → 4×4
        x = self.pool(self.relu(self.conv2(x)))   # 4×4 → 2×2
        x = x.view(x.size(0), -1)                 # 展平
        x = self.fc(x)
        return x


# ====== 加载数据 ======
digits = datasets.load_digits()
X, y = digits.data, digits.target

# CNN 需要 4 维输入：(batch, channel, height, width)
X = X.reshape(-1, 1, 8, 8).astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_tr = torch.tensor(X_tr)
y_tr = torch.tensor(y_tr, dtype=torch.long)
X_te = torch.tensor(X_te)
y_te = torch.tensor(y_te, dtype=torch.long)

# ====== 训练 ======
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    logits = model(X_tr)
    loss = criterion(logits, y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
        print(f'epoch {epoch + 1:>3d}  |  loss={loss.item():.4f}  |  acc={acc:.4f}')

# ====== 测试 ======
with torch.no_grad():
    logits = model(X_te)
    y_pred = logits.argmax(dim=1).numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)
    print(f'\n测试精度: {acc:.4f}')
