# Busca rápida de hiperparâmetros dos modelos clássicos numa divisão única 80/20 do train.xlsx.
#
# Divisão: fold 1 do GroupKFold agrupado por texto (src/divisoes.obter_divisao(data, 'agrupada')): 80% treino,
# 20% avaliação, sem cópias de um mesmo texto dos dois lados. É o mesmo fold 1 das OOF do EP1, então as
# predições do BERTimbau e do ensemble do EP1 nesse fold (feitas por modelos que não o viram) servem de referência.
#
# Cada conjunto de atributos é vetorizado uma vez; os modelos desse conjunto treinam em paralelo.
# Saída: resultados/busca_holdout.csv (uma linha por configuração) e holdout_busca/<config>.npy (probabilidades
# na avaliação; para LinearSVC, NB-SVM e Ridge, softmax dos escores de decisão), para uso posterior no stacking.
# Conjuntos de atributos já concluídos são pulados (a execução pode ser retomada).
#
# Uso: python busca_holdout.py [--atributos w12 w12c25 ...] [--jobs 10]

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.divisoes import obter_divisao

CAMINHO_DADOS = '../train.xlsx'
PASTA_SAIDA = 'holdout_busca'
ARQUIVO_RESULTADOS = 'resultados/busca_holdout.csv'
CLASSES = np.array(['c1', 'c234', 'c5'])
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}


def palavras(ngramas, **kw):
    return {'ngram_range': ngramas, 'sublinear_tf': True, **kw}


def caracteres(ngramas, analyzer='char_wb', max_features=300000, **kw):
    return {'analyzer': analyzer, 'ngram_range': ngramas, 'sublinear_tf': True, 'max_features': max_features, **kw}


# nome -> lista de parâmetros de TfidfVectorizer (concatenados). Conjuntos só de palavras habilitam NB e NB-SVM.
ATRIBUTOS = {
    'w11': [palavras((1, 1))],
    'w12': [palavras((1, 2))],
    'w12_semsub': [palavras((1, 2), sublinear_tf=False)],
    'w12_min2': [palavras((1, 2), min_df=2, max_df=0.9)],
    'w12_semacento': [palavras((1, 2), strip_accents='unicode')],
    'w12_binario': [palavras((1, 2), binary=True)],
    'w13': [palavras((1, 3), min_df=2)],
    'c25': [caracteres((2, 5))],
    'c36': [caracteres((3, 6), max_features=500000)],
    'ch25': [caracteres((2, 5), analyzer='char')],
    'w12c25': [palavras((1, 2)), caracteres((2, 5))],
    'w12c36': [palavras((1, 2)), caracteres((3, 6), max_features=500000)],
    'w13c25': [palavras((1, 3), min_df=2), caracteres((2, 5))],
    'w12c25_semacento': [palavras((1, 2), strip_accents='unicode'), caracteres((2, 5), strip_accents='unicode')],
}


def modelos(so_palavras):
    m = {}
    for C in [0.1, 0.25, 0.5, 1, 2, 4]:
        m[f'logreg_C{C}'] = lambda C=C: LogisticRegression(C=C, max_iter=3000)
    for C in [0.5, 1]:
        m[f'logregbal_C{C}'] = lambda C=C: LogisticRegression(C=C, max_iter=3000, class_weight='balanced')
    for C in [0.02, 0.05, 0.1, 0.2, 0.4]:
        m[f'svc_C{C}'] = lambda C=C: LinearSVC(C=C)
    for a in [1, 3, 10]:
        m[f'ridge_a{a}'] = lambda a=a: RidgeClassifier(alpha=a)
    for a in [1e-5, 3e-5, 1e-4]:
        m[f'sgdhuber_a{a}'] = lambda a=a: SGDClassifier(loss='modified_huber', alpha=a, max_iter=50, tol=None,
                                                        random_state=42)
    if so_palavras:
        for a in [0.1, 0.3, 0.5, 1.0]:
            m[f'cnb_a{a}'] = lambda a=a: ComplementNB(alpha=a)
        for a in [0.03, 0.1, 0.3]:
            m[f'mnb_a{a}'] = lambda a=a: MultinomialNB(alpha=a)
    return m


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def escores(modelo, M):
    if hasattr(modelo, 'predict_proba') and not isinstance(modelo, SGDClassifier):
        return modelo.predict_proba(M)
    return softmax(modelo.decision_function(M))


def treinar(nome, criar, M_tr, y_tr, M_av):
    warnings.filterwarnings('ignore')
    inicio = time.time()
    modelo = criar().fit(M_tr, y_tr)
    return nome, escores(modelo, M_av), time.time() - inicio


def treinar_nbsvm(nome, C, Xb_tr, y_tr, Xb_av):
    """NB-SVM (Wang & Manning, 2012) um-contra-todos: LogReg sobre atributos binários escalados pela razão de
    log-contagens do Naive Bayes de cada classe."""
    warnings.filterwarnings('ignore')
    inicio, cols = time.time(), []
    for c in CLASSES:
        pos = np.asarray(Xb_tr[y_tr == c].sum(0)).ravel() + 1
        neg = np.asarray(Xb_tr[y_tr != c].sum(0)).ravel() + 1
        r = np.log((pos / pos.sum()) / (neg / neg.sum()))
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xb_tr.multiply(r).tocsr(), (y_tr == c).astype(int))
        cols.append(clf.decision_function(Xb_av.multiply(r).tocsr()))
    return nome, softmax(np.column_stack(cols)), time.time() - inicio


def referencias(y_av, idx_av):
    """Acurácia no mesmo fold do BERTimbau e do ensemble do EP1 (OOF do fold 1)."""
    probs = {}
    for nome in PESOS_EP1:
        df = pd.read_csv(f'oof/oof_{nome}.csv').sort_values('indice').reset_index(drop=True)
        probs[nome] = df[[f'p_{c}' for c in CLASSES]].to_numpy()[idx_av]
    linhas = [('ref_bertimbau (EP1)', probs['bertimbau']),
              ('ref_svc_word12_char25 (EP1)', probs['svc_word12_char25']),
              ('ref_logreg_word12 (EP1)', probs['logreg_word12']),
              ('ref_ensemble_entregue (EP1)', sum(w * probs[n] for n, w in PESOS_EP1.items()))]
    for nome, p in linhas:
        prev = CLASSES[p.argmax(1)]
        print(f'{nome:<32} acc {accuracy_score(y_av, prev):.4f}  F1 {f1_score(y_av, prev, average="macro"):.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--atributos', nargs='+', default=list(ATRIBUTOS))
    p.add_argument('--jobs', type=int, default=10)
    args = p.parse_args()

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    os.makedirs(os.path.dirname(ARQUIVO_RESULTADOS), exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()
    idx_tr, idx_av = obter_divisao(data, 'agrupada')
    y_tr, y_av = y[idx_tr], y[idx_av]
    np.save(os.path.join(PASTA_SAIDA, '_indices_avaliacao.npy'), idx_av)
    print(f'Divisão agrupada 80/20: {len(idx_tr)} treino, {len(idx_av)} avaliação\n')
    referencias(y_av, idx_av)

    feitos = pd.read_csv(ARQUIVO_RESULTADOS) if os.path.exists(ARQUIVO_RESULTADOS) else pd.DataFrame()
    for nome_atr in args.atributos:
        if len(feitos) and (feitos['atributos'] == nome_atr).any():
            print(f'\n{nome_atr}: já concluído, pulando', flush=True)
            continue
        inicio = time.time()
        params = ATRIBUTOS[nome_atr]
        vets = [TfidfVectorizer(**pv).fit(X[idx_tr]) for pv in params]
        M_tr = hstack([v.transform(X[idx_tr]) for v in vets]).tocsr()
        M_av = hstack([v.transform(X[idx_av]) for v in vets]).tocsr()
        so_palavras = len(params) == 1 and params[0].get('analyzer', 'word') == 'word'

        tarefas = [delayed(treinar)(f'{n}_{nome_atr}', criar, M_tr, y_tr, M_av)
                   for n, criar in modelos(so_palavras).items()]
        if so_palavras:
            pv = params[0]
            cont = CountVectorizer(ngram_range=pv['ngram_range'], min_df=pv.get('min_df', 1), binary=True,
                                   strip_accents=pv.get('strip_accents')).fit(X[idx_tr])
            Xb_tr, Xb_av = cont.transform(X[idx_tr]), cont.transform(X[idx_av])
            tarefas += [delayed(treinar_nbsvm)(f'nbsvm_C{C}_{nome_atr}', C, Xb_tr, y_tr, Xb_av)
                        for C in [0.05, 0.1, 0.25, 0.5]]
        saidas = Parallel(n_jobs=args.jobs)(tarefas)

        linhas = []
        for config, probs, segundos in saidas:
            np.save(os.path.join(PASTA_SAIDA, f'{config}.npy'), probs.astype(np.float32))
            prev = CLASSES[probs.argmax(1)]
            linhas.append({'config': config, 'modelo': config.split('_')[0], 'atributos': nome_atr,
                           'acc': round(accuracy_score(y_av, prev), 4),
                           'f1_macro': round(f1_score(y_av, prev, average='macro'), 4),
                           'n_atributos': M_tr.shape[1], 'segundos': round(segundos, 1)})
        novos = pd.DataFrame(linhas)
        feitos = pd.concat([feitos, novos], ignore_index=True)
        feitos.to_csv(ARQUIVO_RESULTADOS, index=False)
        top = novos.sort_values('acc', ascending=False).head(3)
        print(f'\n{nome_atr} ({M_tr.shape[1]} atributos): {len(novos)} configs em {time.time() - inicio:.0f}s\n'
              + top[['config', 'acc', 'f1_macro']].to_string(index=False), flush=True)

    print('\n=== Top 20 geral ===')
    print(feitos.sort_values('acc', ascending=False).head(20)[['config', 'acc', 'f1_macro']].to_string(index=False))
    print('\n=== Melhor configuração de cada modelo ===')
    melhor = feitos.loc[feitos.groupby('modelo')['acc'].idxmax()].sort_values('acc', ascending=False)
    print(melhor[['modelo', 'config', 'acc', 'f1_macro']].to_string(index=False))


if __name__ == '__main__':
    main()
