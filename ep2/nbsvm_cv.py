# Validação cruzada de 5 folds do NB-SVM (componente clássico da versão E), nos mesmos folds das OOF do EP1.
# Saída: oof/oof_nbsvm.csv (mesmo formato das OOF do EP1).
#
# Uso: python nbsvm_cv.py

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from busca_holdout import CLASSES, treinar_nbsvm
from nbsvm_final import C, vetorizador
from src.data_loader import load_data
from src.divisoes import folds_cv


def fold(X, y, tr, av):
    v = vetorizador().fit(X[tr])
    return av, treinar_nbsvm('cv', C, v.transform(X[tr]), y[tr], v.transform(X[av]))[1]


def main():
    data = load_data('../train.xlsx')
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()
    folds = folds_cv(data)
    oof, nfold = np.zeros((len(y), 3)), np.zeros(len(y), dtype=int)
    for f, (av, p) in enumerate(Parallel(n_jobs=5)(delayed(fold)(X, y, tr, av) for tr, av in folds), 1):
        oof[av], nfold[av] = p, f
    ref = pd.read_csv('oof/oof_bertimbau.csv').sort_values('indice')['fold'].to_numpy()
    assert (nfold == ref).all(), 'folds diferentes das OOF do EP1'
    df = pd.DataFrame(oof, columns=[f'p_{c}' for c in CLASSES])
    df.insert(0, 'fold', nfold); df.insert(0, 'indice', np.arange(len(y)))
    df['rotulo'], df['previsto'] = y, CLASSES[oof.argmax(1)]
    df.to_csv('oof/oof_nbsvm.csv', index=False)
    print('acurácia por fold:', [round(np.mean(df.previsto[nfold == f] == y[nfold == f]), 4) for f in range(1, 6)],
          f'| total {np.mean(df.previsto == y):.4f}')


if __name__ == '__main__':
    main()
