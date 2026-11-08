# NB-SVM final (componente clássico da versão final): treina em 100% de dados/train.xlsx e grava as probabilidades
# do test1 e do test2 em saidas/nbsvm/final_<teste>.csv.
#
# Uso: python nbsvm_final.py [--conferir]
#   --conferir refaz o holdout 80/20 (fold 1) e compara com as probabilidades salvas em saidas/transformers_holdout.

import argparse
import os

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.divisoes import obter_divisao
from src.nbsvm import CLASSES, NBSVM

PASTA = 'saidas/nbsvm'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--conferir', action='store_true')
    args = p.parse_args()
    data = load_data('dados/train.xlsx')
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()

    if args.conferir:
        tr, av = obter_divisao(data, 'agrupada')
        probs = NBSVM(X[tr], y[tr]).predict_proba(X[av])
        ref = np.load('saidas/transformers_holdout/nbsvm_C0.25_w12_semacento.npy')
        print(f'holdout: acc {np.mean(CLASSES[probs.argmax(1)] == y[av]):.4f} '
              f'| igual ao salvo: {np.allclose(probs, ref, atol=1e-4)}')

    modelo = NBSVM(X, y)
    os.makedirs(PASTA, exist_ok=True)
    for nome in ['test1', 'test2']:
        teste = pd.read_excel(f'dados/{nome}.xlsx')
        probs = modelo.predict_proba(teste['resp_text'].fillna('').astype(str).to_numpy())
        df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
        df.insert(0, 'indice', np.arange(len(teste)))
        df['previsto'] = CLASSES[probs.argmax(1)]
        df.to_csv(os.path.join(PASTA, f'final_{nome}.csv'), index=False)
        print(f'{nome}: {len(teste)} textos | {df["previsto"].value_counts().to_dict()} -> {PASTA}/final_{nome}.csv')


if __name__ == '__main__':
    main()
