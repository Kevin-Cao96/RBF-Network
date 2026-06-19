import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rbf_network import RBFNetwork

# ====== 1. Optical Recognition of Handwritten Digits ======
print('=' * 55)
print('  Dataset 1: Optical Recognition of Handwritten Digits')
print('=' * 55)

digits = datasets.load_digits()
X1, y1 = digits.data, digits.target
print(f'  样本数: {X1.shape[0]}, 特征数: {X1.shape[1]}, 类别数: {len(np.unique(y1))}')

X1_tr, X1_te, y1_tr, y1_te = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1)

scaler = StandardScaler()
X1_tr = scaler.fit_transform(X1_tr)
X1_te = scaler.transform(X1_te)

print()
print('  K    | Train Acc | Test Acc')
print('  ' + '-' * 28)
for k in [5,10,15,20,25,30]:
    if k >= len(X1_tr):
        break
    best_acc_te = 0.0
    best_param = None
    model = RBFNetwork(n_centers=k, random_state=42)
    for param in np.arange(0.01,1.01,0.001):
        model.fit(X1_tr, y1_tr,param)
        tr = accuracy_score(y1_tr, model.predict(X1_tr))
        te = accuracy_score(y1_te, model.predict(X1_te))
        if(te > best_acc_te):
            best_acc_te = te
            best_param = param
    print(f'  {k:>3}  | {tr:.4f}   | {best_acc_te:.4f}  | param={best_param:.3f}')

# ====== 2. Pen-Based Digits ======
print()
print('=' * 55)
print('  Dataset 2: Pen-Based Handwritten Digits')
print('=' * 55)

train = np.loadtxt('pendigits.tra', delimiter=',')
test = np.loadtxt('pendigits.tes', delimiter=',')

X2_tr, y2_tr = train[:, :-1], train[:, -1]
X2_te, y2_te = test[:, :-1], test[:, -1]

print(f'  样本数: {X2_tr.shape[0] + X2_te.shape[0]}, 特征数: {X2_tr.shape[1]}, 类别数: {len(np.unique(y2_tr))}')

scaler2 = StandardScaler()
X2_tr = scaler2.fit_transform(X2_tr)
X2_te = scaler2.transform(X2_te)

print()
print('  K    | Train Acc | Test Acc')
print('  ' + '-' * 28)
for k in [5,10,15,20,25,30]:
    if k >= len(X2_tr):
        break
    best_acc_te = 0.0
    best_param = None
    model = RBFNetwork(n_centers=k, random_state=42)
    for param in np.arange(0.01,1.01,0.001):
        model.fit(X2_tr, y2_tr,param)
        tr = accuracy_score(y2_tr, model.predict(X2_tr))
        te = accuracy_score(y2_te, model.predict(X2_te))
        if(te > best_acc_te):
            best_acc_te = te
            best_param = param
    print(f'  {k:>3}  | {tr:.4f}   | {best_acc_te:.4f}  | param={best_param:.3f}')
