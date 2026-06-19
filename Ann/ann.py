import numpy as np

class ANN:
    """单隐藏层全连接神经网络"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def train(self, X, y, epochs):
        n = X.shape[0]
        y_one_hot = np.eye(self.W2.shape[1])[y]

        for epoch in range(epochs):
            logits = self.forward(X)
            probs = self.softmax(logits)

            dz2 = probs - y_one_hot
            dW2 = self.a1.T @ dz2 / n
            db2 = np.sum(dz2, axis=0, keepdims=True) / n

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (self.z1 > 0)
            dW1 = X.T @ dz1 / n
            db1 = np.sum(dz1, axis=0, keepdims=True) / n

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if (epoch + 1) % 100 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
                print(f'epoch {epoch + 1:>4d}  |  loss={loss:.4f}  |  acc={acc:.4f}')

    def predict(self, X):
        logits = self.forward(X)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = ANN(input_size=64, hidden_size=32, output_size=10, learning_rate=0.1)
    model.train(X_tr, y_tr, epochs=500)

    y_pred = model.predict(X_te)
    print(f'\n测试精度: {accuracy_score(y_te, y_pred):.4f}')
