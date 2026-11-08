# NB-SVM (Wang e Manning, 2012), componente clássico da versão final.
# Configuração escolhida na busca de hiperparâmetros (holdout 80/20):
#   CountVectorizer binário, palavras 1-2, sem acentos; regressão logística C = 0,25, um-contra-todos sobre os
#   atributos escalados pela razão de log-contagem do Naive Bayes de cada classe.

import warnings

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

CLASSES = np.array(['c1', 'c234', 'c5'])
C = 0.25


def vetorizador():
    return CountVectorizer(ngram_range=(1, 2), binary=True, strip_accents='unicode')


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NBSVM:
    """Treina o vetorizador e os 3 classificadores um-contra-todos; guarda os pesos para a interpretação."""

    def __init__(self, X, y, C=C):
        warnings.filterwarnings('ignore')
        self.vet = vetorizador().fit(X)
        B = self.vet.transform(X)
        self.r, self.clfs = [], []
        for c in CLASSES:
            pos = np.asarray(B[y == c].sum(0)).ravel() + 1
            neg = np.asarray(B[y != c].sum(0)).ravel() + 1
            r = np.log((pos / pos.sum()) / (neg / neg.sum()))
            self.r.append(r)
            self.clfs.append(LogisticRegression(C=C, max_iter=3000).fit(B.multiply(r).tocsr(), (y == c).astype(int)))
        self.nomes = self.vet.get_feature_names_out()
        self.pesos = np.vstack([clf.coef_.ravel() * r for clf, r in zip(self.clfs, self.r)])   # 3 x n_atributos

    def predict_proba(self, textos):
        B = self.vet.transform(list(textos))
        return softmax(np.column_stack([clf.decision_function(B.multiply(r).tocsr())
                                        for clf, r in zip(self.clfs, self.r)]))

    def contribuicoes(self, texto, k):
        """Contribuição de cada n-grama presente no texto para a classe k (atributos binários: contribuição = peso)."""
        idx = self.vet.transform([texto]).nonzero()[1]
        return sorted(((self.nomes[j], self.pesos[k, j]) for j in idx), key=lambda t: -t[1])
