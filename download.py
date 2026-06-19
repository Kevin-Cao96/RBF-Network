from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rbf_network import RBFNetwork

# 1. 下载数据集（第一次会下载，之后会缓存）
print("下载 Pendigits 数据集...")
pendigits = fetch_openml('pendigits', parser='auto')
X = pendigits.data.values.astype(float)
y = pendigits.target.values.astype(int)

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(set(y))}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 训练 RBF 网络
for k in [10, 30, 50, 80, 100, 150, 200]:
    if k >= len(X_train):
        break
    model = RBFNetwork(n_centers=k, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"K={k:>3}  |  Test Acc = {acc:.4f}")