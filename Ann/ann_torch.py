import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ====== 定义网络 ======
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ====== 加载数据 ======
digits = datasets.load_digits()
X, y = digits.data, digits.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

X_tr = torch.tensor(X_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.long)
X_te = torch.tensor(X_te, dtype=torch.float32)
y_te = torch.tensor(y_te, dtype=torch.long)

# ====== 训练 ======
model = ANN(input_size=64, hidden_size=32, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    logits = model(X_tr)
    loss = criterion(logits, y_tr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            acc = (probs.argmax(dim=1) == y_tr).float().mean().item()
        print(f'epoch {epoch + 1:>4d}  |  loss={loss.item():.4f}  |  acc={acc:.4f}')

# ====== 测试 ======
with torch.no_grad():
    logits = model(X_te)
    y_pred = logits.argmax(dim=1).numpy()
    acc = accuracy_score(y_te.numpy(), y_pred)
    print(f'\n测试精度: {acc:.4f}')
