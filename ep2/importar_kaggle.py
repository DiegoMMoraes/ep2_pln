# Importa as probabilidades do holdout geradas no Kaggle/Colab (kaggle_bertimbau_large/bertimbau_large.ipynb)
# para holdout_busca/<nome>.npy, na mesma ordem dos índices de avaliação usados pelos clássicos, e mostra a
# acurácia. Depois disso o modelo pode entrar no stacking:
#   python stacking_holdout.py --modelos <nome> bertimbau logreg_C0.25_w12c25 ...
#
# Uso: python importar_kaggle.py <holdout_probabilidades_epoca_N.csv> --nome bertimbau_large

import argparse
import os

import numpy as np
import pandas as pd

CLASSES = np.array(['c1', 'c234', 'c5'])
PASTA = 'holdout_busca'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arquivo')
    p.add_argument('--nome', default='bertimbau_large')
    args = p.parse_args()

    idx_av = np.load(os.path.join(PASTA, '_indices_avaliacao.npy'))
    df = pd.read_csv(args.arquivo).set_index('indice')
    assert set(df.index) == set(idx_av), 'os índices do arquivo não são os da divisão 80/20'
    df = df.loc[idx_av]
    probs = df[[f'p_{c}' for c in CLASSES]].to_numpy(dtype=np.float32)
    ref = pd.read_csv('oof/oof_bertimbau.csv').sort_values('indice')['rotulo'].to_numpy()[idx_av]
    assert (df['rotulo'].to_numpy() == ref).all(), 'rótulos diferentes do train.xlsx'

    np.save(os.path.join(PASTA, f'{args.nome}.npy'), probs)
    previsto = CLASSES[probs.argmax(1)]
    print(f'{args.nome}: acurácia {np.mean(previsto == ref):.4f} em {len(ref)} textos | distribuição prevista '
          + str(pd.Series(previsto).value_counts().to_dict()))
    print(f'-> {PASTA}/{args.nome}.npy')


if __name__ == '__main__':
    main()
