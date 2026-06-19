import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from rbf_network import RBFNetwork

# 加载 Fashion-MNIST
data = np.load('rbf-network-benchmark/data/fashion_mnist.npz', allow_pickle=False)
X, y = data['x'], data['y']

# 子采样（70000 全量跑太慢，取 8000 快速演示）
rng = np.random.RandomState(42)
idx = rng.choice(len(X), 8000, replace=False)
X, y = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# 标准化 + PCA 降维
X_train = X_train / 255.0
X_test = X_test / 255.0
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=50, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(f'数据: {X_train.shape[1]} 特征 (PCA), {X_train.shape[0]} 训练, {X_test.shape[0]} 测试')
print()

for k in [50, 100, 200, 400]:
    model = RBFNetwork(n_centers=k, random_state=42)
    model.fit(X_train, y_train)
    tr = accuracy_score(y_train, model.predict(X_train))
    te = accuracy_score(y_test, model.predict(X_test))
    print(f'K={k:>3}  | Train={tr:.4f}  | Test={te:.4f}')
