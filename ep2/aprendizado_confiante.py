# Aprendizado confiante (Northcutt et al., 2021; biblioteca cleanlab) para lidar com o ruído dos rótulos.
#
# 1. Dentro dos 80% de treino da divisão agrupada, gera probabilidades out-of-fold com validação cruzada interna
#    (5 folds agrupados por texto), usando a média de LogReg (palavras 1-2 + caracteres 2-5) e NB-SVM.
#    Os 20% de avaliação não participam de nada nesta etapa (sem vazamento).
# 2. cleanlab.find_label_issues aponta os exemplos de treino com rótulo provavelmente errado.
# 3. Treina LogReg e NB-SVM nos 80% (a) originais, (b) sem os exemplos suspeitos, (c) com os suspeitos
#    re-rotulados para a classe prevista e (d) com peso reduzido nos suspeitos; avalia nos 20% (rótulos originais).
#
# Uso: python aprendizado_confiante.py

import os
import warnings

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from busca_holdout import CLASSES, softmax
from src.data_loader import load_data
from src.deduplicacao import chave_texto
from src.divisoes import obter_divisao

warnings.filterwarnings('ignore')


def tfidf_w12c25(X_tr, X_av):
    vets = [TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(X_tr),
            TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, max_features=300000).fit(X_tr)]
    return (hstack([v.transform(X_tr) for v in vets]).tocsr(), hstack([v.transform(X_av) for v in vets]).tocsr())


def logreg(X_tr, y_tr, X_av, peso=None):
    M_tr, M_av = tfidf_w12c25(X_tr, X_av)
    return LogisticRegression(C=0.25, max_iter=3000).fit(M_tr, y_tr, sample_weight=peso).predict_proba(M_av)


def nbsvm(X_tr, y_tr, X_av, peso=None, C=0.25):
    cont = CountVectorizer(ngram_range=(1, 2), binary=True, strip_accents='unicode').fit(X_tr)
    B_tr, B_av = cont.transform(X_tr), cont.transform(X_av)
    peso = np.ones(len(y_tr)) if peso is None else peso
    cols = []
    for c in CLASSES:
        pos = np.asarray(B_tr[y_tr == c].multiply(peso[y_tr == c][:, None]).sum(0)).ravel() + 1
        neg = np.asarray(B_tr[y_tr != c].multiply(peso[y_tr != c][:, None]).sum(0)).ravel() + 1
        r = np.log((pos / pos.sum()) / (neg / neg.sum()))
        clf = LogisticRegression(C=C, max_iter=3000).fit(B_tr.multiply(r).tocsr(), (y_tr == c).astype(int),
                                                          sample_weight=peso)
        cols.append(clf.decision_function(B_av.multiply(r).tocsr()))
    return softmax(np.column_stack(cols))


def main():
    data = load_data('../train.xlsx')
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()
    idx_tr, idx_av = obter_divisao(data, 'agrupada')
    X_tr, y_tr, X_av, y_av = X[idx_tr], y[idx_tr], X[idx_av], y[idx_av]
    y_tr_num = pd.Series(y_tr).map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # 1. probabilidades out-of-fold dentro do treino
    cache = 'resultados/confiante_oof_treino.npy'
    if os.path.exists(cache):
        oof = np.load(cache)
    else:
        oof = np.zeros((len(y_tr), 3))
        grupos = pd.Series(X_tr).apply(chave_texto)
        for f, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=7).split(X_tr, groups=grupos), 1):
            oof[b] = (logreg(X_tr[a], y_tr[a], X_tr[b]) + nbsvm(X_tr[a], y_tr[a], X_tr[b])) / 2
            print(f'  OOF fold {f}/5', flush=True)
        np.save(cache, oof)
    print(f'Acurácia OOF dentro do treino: {np.mean(CLASSES[oof.argmax(1)] == y_tr):.4f}')

    # 2. rótulos suspeitos
    suspeitos = find_label_issues(y_tr_num, oof, filter_by='prune_by_noise_rate', return_indices_ranked_by=None)
    print(f'Suspeitos: {suspeitos.sum()} de {len(y_tr)} ({suspeitos.mean():.1%}) | por classe rotulada: '
          + str(pd.Series(y_tr[suspeitos]).value_counts().reindex(CLASSES).to_dict()))

    # 3. treina e avalia
    variantes = {
        'original': (X_tr, y_tr, None),
        'sem suspeitos': (X_tr[~suspeitos], y_tr[~suspeitos], None),
        're-rotulado': (X_tr, np.where(suspeitos, CLASSES[oof.argmax(1)], y_tr), None),
        'peso 0,5 nos suspeitos': (X_tr, y_tr, np.where(suspeitos, 0.5, 1.0)),
    }
    linhas = []
    for nome, (Xv, yv, peso) in variantes.items():
        p_lr, p_nb = logreg(Xv, yv, X_av, peso), nbsvm(Xv, yv, X_av, peso)
        acc = {m: np.mean(CLASSES[p.argmax(1)] == y_av) for m, p in
               [('LogReg', p_lr), ('NB-SVM', p_nb), ('média LogReg+NB-SVM', p_lr + p_nb)]}
        linhas.append({'variante': nome, 'n_treino': len(yv), **{k: round(v, 4) for k, v in acc.items()}})
        print(f'  {nome:<24} ' + ' | '.join(f'{k} {v:.4f}' for k, v in acc.items()), flush=True)
        if nome != 'original':
            np.save(f'holdout_busca/nbsvm_confiante_{nome.split()[0]}.npy', p_nb.astype(np.float32))
    pd.DataFrame(linhas).to_csv('resultados/aprendizado_confiante.csv', index=False)


if __name__ == '__main__':
    main()
