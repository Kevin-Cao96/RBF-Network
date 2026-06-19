import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer


class RBFNetwork:

    def __init__(self, n_centers=10, sigma=None, random_state=42, ridge_alpha=1e-6):
        self.n_centers = n_centers
        self.sigma = sigma
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.centers_ = None
        self.sigma_ = None
        self.weights_ = None

    def _gaussian(self, X):
        n_samples = X.shape[0]
        phi = np.zeros((n_samples, self.n_centers), dtype=np.float64)
        for j in range(self.n_centers):
            diff = X - self.centers_[j]
            dist_sq = np.sum(diff ** 2, axis=1)
            phi[:, j] = np.exp(-dist_sq / (2.0 * self.sigma_ ** 2))
        return phi

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)

        kmeans = KMeans(n_clusters=self.n_centers,
                        random_state=self.random_state,
                        n_init='auto')
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_

        if self.sigma is None:
            pairwise_dist = np.zeros((self.n_centers, self.n_centers))
            for i in range(self.n_centers):
                diff = self.centers_ - self.centers_[i]
                pairwise_dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))
            d_max = pairwise_dist.max()
            self.sigma_ = d_max / np.sqrt(2.0 * self.n_centers)
            if self.sigma_ < 1e-12:
                self.sigma_ = 1.0
        else:
            self.sigma_ = self.sigma

        Phi = self._gaussian(X)

        self.binarizer_ = LabelBinarizer().fit(y)
        Y = self.binarizer_.transform(y)

        A = Phi.T @ Phi + self.ridge_alpha * np.eye(self.n_centers)
        self.weights_ = np.linalg.solve(A, Phi.T @ Y)

        return self

    def predict(self, X):
        Phi = self._gaussian(X)
        scores = Phi @ self.weights_
        return self.binarizer_.inverse_transform(scores)
print('saved')
