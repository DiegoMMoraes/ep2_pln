# Busca de hiperparâmetros dos modelos clássicos (LogReg, LinearSVC, NB-SVM, ComplementNB, MultinomialNB)
# com validação cruzada agrupada de 5 folds (os mesmos folds das OOF do EP1, src/divisoes.folds_cv).
#
# Para economizar tempo, cada fold vetoriza o texto uma vez por conjunto de atributos e treina todos os modelos
# desse conjunto; os 5 folds rodam em paralelo.
# Para cada configuração, grava a acurácia em resultados/busca_classicos.csv e os escores out-of-fold em
# oof_busca/<config>.npy (probabilidades; para o LinearSVC e o NB-SVM, softmax dos escores de decisão), que
# podem ser usados no stacking. Conjuntos de atributos já concluídos são pulados (a execução pode ser retomada).
#
# Uso: python busca_classicos.py [--atributos w12 w12c25 ...]

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.divisoes import folds_cv

CAMINHO_DADOS = '../train.xlsx'
PASTA_OOF = 'oof_busca'
ARQUIVO_RESULTADOS = 'resultados/busca_classicos.csv'
CLASSES = np.array(['c1', 'c234', 'c5'])


def palavras(ngramas, min_df=1):
    return lambda: TfidfVectorizer(ngram_range=ngramas, sublinear_tf=True, min_df=min_df)


def caracteres(ngramas, max_features=300000):
    return lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=ngramas, sublinear_tf=True,
                                   max_features=max_features)


# nome -> lista de vetorizadores (concatenados) e se o conjunto é só de palavras (habilita os Naive Bayes e NB-SVM)
ATRIBUTOS = {
    'w11': ([palavras((1, 1))], True),
    'w12': ([palavras((1, 2))], True),
    'w13': ([palavras((1, 3), min_df=2)], True),
    'c25': ([caracteres((2, 5))], False),
    'c36': ([caracteres((3, 6), max_features=500000)], False),
    'w12c25': ([palavras((1, 2)), caracteres((2, 5))], False),
    'w13c25': ([palavras((1, 3), min_df=2), caracteres((2, 5))], False),
    'w12c36': ([palavras((1, 2)), caracteres((3, 6), max_features=500000)], False),
}

C_LOGREG = [0.1, 0.25, 0.5, 1, 2, 4, 8]
C_SVC = [0.02, 0.05, 0.1, 0.2, 0.4]
C_NBSVM = [0.05, 0.1, 0.25, 0.5, 1]
ALPHA_CNB = [0.1, 0.3, 0.5, 1.0]
ALPHA_MNB = [0.03, 0.1, 0.3]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nbsvm_escores(Xb_tr, y_tr, Xb_val, C):
    """NB-SVM (Wang & Manning, 2012) um-contra-todos: LogReg sobre atributos binários escalados pela razão
    de log-contagens do Naive Bayes de cada classe."""
    escores = []
    for c in CLASSES:
        pos = np.asarray(Xb_tr[y_tr == c].sum(0)).ravel() + 1
        neg = np.asarray(Xb_tr[y_tr != c].sum(0)).ravel() + 1
        r = np.log((pos / pos.sum()) / (neg / neg.sum()))
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xb_tr.multiply(r).tocsr(), (y_tr == c).astype(int))
        escores.append(clf.decision_function(Xb_val.multiply(r).tocsr()))
    return np.column_stack(escores)


def rodar_fold(nome_atr, X, y, idx_tr, idx_val):
    warnings.filterwarnings('ignore')
    vetorizadores, so_palavras = ATRIBUTOS[nome_atr]
    vets = [criar().fit(X[idx_tr]) for criar in vetorizadores]
    M_tr = hstack([v.transform(X[idx_tr]) for v in vets]).tocsr()
    M_val = hstack([v.transform(X[idx_val]) for v in vets]).tocsr()
    y_tr = y[idx_tr]
    saida = {}

    for C in C_LOGREG:
        saida[f'logreg_{nome_atr}_C{C}'] = LogisticRegression(C=C, max_iter=3000).fit(M_tr, y_tr).predict_proba(M_val)
    for C in C_SVC:
        saida[f'svc_{nome_atr}_C{C}'] = softmax(LinearSVC(C=C).fit(M_tr, y_tr).decision_function(M_val))

    if so_palavras:
        for a in ALPHA_CNB:
            saida[f'cnb_{nome_atr}_a{a}'] = ComplementNB(alpha=a).fit(M_tr, y_tr).predict_proba(M_val)
        for a in ALPHA_MNB:
            saida[f'mnb_{nome_atr}_a{a}'] = MultinomialNB(alpha=a).fit(M_tr, y_tr).predict_proba(M_val)
        cont = CountVectorizer(ngram_range=vets[0].ngram_range, min_df=vets[0].min_df, binary=True).fit(X[idx_tr])
        Xb_tr, Xb_val = cont.transform(X[idx_tr]), cont.transform(X[idx_val])
        for C in C_NBSVM:
            saida[f'nbsvm_{nome_atr}_C{C}'] = softmax(nbsvm_escores(Xb_tr, y_tr, Xb_val, C))
    return idx_val, saida


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--atributos', nargs='+', default=list(ATRIBUTOS))
    p.add_argument('--jobs', type=int, default=5)
    args = p.parse_args()

    os.makedirs(PASTA_OOF, exist_ok=True)
    os.makedirs(os.path.dirname(ARQUIVO_RESULTADOS), exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()
    folds = folds_cv(data)
    numero_fold = np.empty(len(y), dtype=int)
    for f, (_, idx_val) in enumerate(folds, start=1):
        numero_fold[idx_val] = f
    # confere que os folds são os mesmos das OOF do EP1 (necessário para o stacking)
    ref = pd.read_csv('oof/oof_bertimbau.csv').sort_values('indice')
    assert (ref['fold'].to_numpy() == numero_fold).all(), 'folds diferentes dos usados nas OOF do EP1'

    feitos = pd.read_csv(ARQUIVO_RESULTADOS) if os.path.exists(ARQUIVO_RESULTADOS) else pd.DataFrame()
    for nome_atr in args.atributos:
        if len(feitos) and (feitos['atributos'] == nome_atr).any():
            print(f'{nome_atr}: já concluído, pulando', flush=True)
            continue
        inicio = time.time()
        partes = Parallel(n_jobs=args.jobs)(delayed(rodar_fold)(nome_atr, X, y, tr, val) for tr, val in folds)
        linhas = []
        for config in partes[0][1]:
            oof = np.zeros((len(y), len(CLASSES)))
            for idx_val, saida in partes:
                oof[idx_val] = saida[config]
            np.save(os.path.join(PASTA_OOF, f'{config}.npy'), oof.astype(np.float32))
            previsto = CLASSES[oof.argmax(1)]
            accs = [accuracy_score(y[numero_fold == f], previsto[numero_fold == f]) for f in range(1, 6)]
            linhas.append({'config': config, 'modelo': config.split('_')[0], 'atributos': nome_atr,
                           'acc': round(accuracy_score(y, previsto), 4), 'acc_dp_folds': round(np.std(accs), 4),
                           'f1_macro': round(f1_score(y, previsto, average='macro'), 4)})
        novos = pd.DataFrame(linhas)
        feitos = pd.concat([feitos, novos], ignore_index=True)
        feitos.to_csv(ARQUIVO_RESULTADOS, index=False)
        melhor = novos.sort_values('acc', ascending=False).head(3)
        print(f'{nome_atr}: {len(novos)} configs em {time.time() - inicio:.0f}s; melhores:\n'
              + melhor[['config', 'acc', 'acc_dp_folds', 'f1_macro']].to_string(index=False), flush=True)

    print('\nTop 15 geral:')
    print(feitos.sort_values('acc', ascending=False).head(15).to_string(index=False))


if __name__ == '__main__':
    main()
