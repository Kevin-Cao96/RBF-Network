import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rbf_network import RBFNetwork


# ------------------------------------------------------------------
#  Load datasets
# ------------------------------------------------------------------

def load_data(name):
    if name == 'Iris':
        X, y = datasets.load_iris(return_X_y=True)
    elif name == 'Wine':
        X, y = datasets.load_wine(return_X_y=True)
    elif name == 'Breast Cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return X, y


# ------------------------------------------------------------------
#  Train & evaluate with different center counts
# ------------------------------------------------------------------

DATASETS = ['Iris', 'Wine', 'Breast Cancer']
N_CENTERS = list(range(10, 101, 10))       # 10, 20, ..., 100

results = {name: [] for name in DATASETS}  # results[name] = [(k, acc), ...]

for name in DATASETS:
    print(f'[{name}]')
    X, y = load_data(name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for k in N_CENTERS:
        model = RBFNetwork(n_centers=k, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name].append((k, acc))
        print(f'  K={k:>3}  acc={acc:.4f}')
    print()


# ------------------------------------------------------------------
#  Plot
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for name in DATASETS:
    ks = [r[0] for r in results[name]]
    accs = [r[1] for r in results[name]]
    ax.plot(ks, accs, marker='o', label=name)

ax.set_xlabel('Number of RBF centers (K)')
ax.set_ylabel('Test Accuracy')
ax.set_title('RBF Network: Accuracy vs Number of Centers')
ax.set_xticks(N_CENTERS)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('/Users/yijiacao/Documents/纳米所/accuracy_plot.png', dpi=150)
print('Saved: accuracy_plot.png')
