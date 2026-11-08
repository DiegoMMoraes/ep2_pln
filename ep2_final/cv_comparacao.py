# Compara, na validação cruzada de 5 folds (20.092 textos, mesmos folds do EP1), os 3 melhores modelos isolados
# e a versão final com o ensemble entregue no EP1.
#
# OOF de cada transformer = fold 1 (holdout 80/20, saidas/transformers_holdout/<nome>.npy) + folds 2-5 (Kaggle, modo cv,
# saidas/transformers_cv/cv_<nome>_f*/cv_fold<f>_semente<s>_epoca<última>.csv).
# NB-SVM: saidas/oof/oof_nbsvm.csv (nbsvm_cv.py). EP1: saidas/oof/oof_{bertimbau,svc_word12_char25,logreg_word12}.csv.
#
# Uso: python cv_comparacao.py

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CLASSES = np.array(['c1', 'c234', 'c5'])
PASTA_CV = 'saidas/transformers_cv'
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}
TRANSFORMERS = {'bertimbau_large': 42, 'bertimbau_large_s43': 43, 'bertimbau_large_suave': 42, 'est_ema_s43': 43,
                'est_ema_llrd_3ep_s42': 42, 'xlmr_large': 42, 'xlmr_large_suave': 42}
NOMES = {'bertimbau_large_suave': 'BERTimbau large + rótulos suaves', 'est_ema_s43': 'BERTimbau large + EMA (s43)',
         'est_ema_llrd_3ep_s42': 'BERTimbau large + EMA+LLRD 3 épocas'}
VERSAO_E = ['bertimbau_large_suave', 'bertimbau_large', 'bertimbau_large_s43', 'xlmr_large', 'xlmr_large_suave',
            'nbsvm']


def oof_csv(nome):
    d = pd.read_csv(f'saidas/oof/oof_{nome}.csv').sort_values('indice')
    return d[[f'p_{c}' for c in CLASSES]].to_numpy(), d['rotulo'].to_numpy(), d['fold'].to_numpy()


def oof_transformer(nome, semente, n, fold):
    probs = np.full((n, 3), np.nan)
    idx1 = np.load('saidas/transformers_holdout/_indices_avaliacao.npy')
    probs[idx1] = np.load(f'saidas/transformers_holdout/{nome}.npy')
    for f in range(2, 6):
        arqs = sorted(glob.glob(os.path.join(PASTA_CV, f'cv_{nome}_f*', f'cv_fold{f}_semente{semente}_epoca*.csv')))
        assert arqs, f'{nome}: fold {f} não encontrado'
        d = pd.read_csv(arqs[-1])                          # última época
        assert (d['fold'] == f).all() and (fold[d['indice']] == f).all(), f'{nome}: fold {f} inconsistente'
        probs[d['indice'].to_numpy()] = d[[f'p_{c}' for c in CLASSES]].to_numpy()
    assert not np.isnan(probs).any(), f'{nome}: linhas sem previsão'
    return probs


def main():
    _, y, fold = oof_csv('bertimbau')
    n = len(y)
    P = {k: oof_transformer(k, s, n, fold) for k, s in TRANSFORMERS.items()}
    P['nbsvm'] = oof_csv('nbsvm')[0]
    ep1 = sum(w * oof_csv(k)[0] for k, w in PESOS_EP1.items())
    acerto_ep1 = CLASSES[ep1.argmax(1)] == y

    def linha(nome, probs):
        acerto = CLASSES[probs.argmax(1)] == y
        por_fold = [acerto[fold == f].mean() for f in range(1, 6)]
        n10, n01 = int((acerto_ep1 & ~acerto).sum()), int((~acerto_ep1 & acerto).sum())
        melhores = sum(a > b for a, b in zip(por_fold, [acerto_ep1[fold == f].mean() for f in range(1, 6)]))
        return {'modelo': nome, 'acc_cv': round(acerto.mean(), 4), 'dp_folds': round(np.std(por_fold), 4),
                **{f'fold{f}': round(v, 4) for f, v in enumerate(por_fold, 1)},
                'vs_EP1': round(acerto.mean() - acerto_ep1.mean(), 4), 'folds_acima_EP1': f'{melhores}/5',
                'so_EP1_acerta': n10, 'so_este_acerta': n01,
                'mcnemar_p': round(binomtest(n01, n10 + n01, 0.5).pvalue, 4) if nome != 'EP1' else None}

    linhas = [linha('Ensemble do EP1 (entregue)', ep1)]
    linhas += [linha(NOMES[k], P[k]) for k in NOMES]
    linhas += [linha('Versão final (6 componentes)', sum(P[k] for k in VERSAO_E))]
    linhas += [linha(f'  componente: {k}', P[k]) for k in VERSAO_E if k not in NOMES]
    tab = pd.DataFrame(linhas)
    pd.set_option('display.width', 250)
    print(f'Validação cruzada de 5 folds agrupados ({n} textos, mesmos folds do EP1)\n')
    print(tab.to_string(index=False))
    tab.to_csv('resultados/cv_comparacao.csv', index=False)
    print('\n-> resultados/cv_comparacao.csv')


if __name__ == '__main__':
    main()
