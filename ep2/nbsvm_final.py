# NB-SVM final (componente clássico da versão E): treina em 100% do train.xlsx e grava as probabilidades do
# test1 e do test2 em finais/nbsvm/final_<teste>.csv, no mesmo formato da saída do Kaggle.
# Configuração escolhida na busca 80/20 (busca_holdout.py): nbsvm_C0.25_w12_semacento
#   = CountVectorizer binário, palavras 1-2, sem acentos; LogReg C=0,25 um-contra-todos sobre a razão do NB.
#
# Uso: python nbsvm_final.py [--conferir]   (--conferir refaz o holdout 80/20 e compara com o valor da busca)

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from busca_holdout import CLASSES, treinar_nbsvm
from src.data_loader import load_data
from src.divisoes import obter_divisao

C = 0.25
PASTA = 'finais/nbsvm'


def vetorizador():
    return CountVectorizer(ngram_range=(1, 2), binary=True, strip_accents='unicode')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--conferir', action='store_true')
    args = p.parse_args()
    data = load_data('../train.xlsx')
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()

    if args.conferir:
        tr, av = obter_divisao(data, 'agrupada')
        v = vetorizador().fit(X[tr])
        _, probs, _ = treinar_nbsvm('conf', C, v.transform(X[tr]), y[tr], v.transform(X[av]))
        ref = np.load('holdout_busca/nbsvm_C0.25_w12_semacento.npy')
        print(f'holdout: acc {np.mean(CLASSES[probs.argmax(1)] == y[av]):.4f} '
              f'| igual à busca: {np.allclose(probs, ref, atol=1e-4)}')

    v = vetorizador().fit(X)
    os.makedirs(PASTA, exist_ok=True)
    for nome in ['test1', 'test2']:
        caminho = '../ep1/data/test1.xlsx' if nome == 'test1' else '../test2.xlsx'   # test1 sem rótulos
        teste = pd.read_excel(caminho)
        textos = teste['resp_text'].fillna('').astype(str).to_numpy()
        _, probs, _ = treinar_nbsvm(nome, C, v.transform(X), y, v.transform(textos))
        df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
        df.insert(0, 'indice', np.arange(len(teste)))
        df['previsto'] = CLASSES[probs.argmax(1)]
        df.to_csv(os.path.join(PASTA, f'final_{nome}.csv'), index=False)
        print(f'{nome}: {len(teste)} textos | {df["previsto"].value_counts().to_dict()} -> {PASTA}/final_{nome}.csv')


if __name__ == '__main__':
    main()
