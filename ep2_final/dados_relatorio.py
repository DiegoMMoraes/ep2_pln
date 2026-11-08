# Números citados no relatório que ainda não tinham saída salva.
#   1. Robustez da validação cruzada: versão final x EP1 só nos folds 2-5 (sem o fold 1, usado na escolha da E) e sem o
#      componente xlmr_large_suave; McNemar exato e qui-quadrado com correção de continuidade.
#   2. Poder estatístico do test1: em amostras balanceadas de 900 textos (300 por classe, como o test1) sorteadas das
#      previsões out-of-fold da CV (20.092 textos), com que frequência o McNemar detecta (p < 0,05) a vantagem da E.
# A decodificação balanceada é registrada à parte: python decodificacao_balanceada.py > resultados/log_decodificacao_balanceada.txt
#
# Uso: python dados_relatorio.py   (saída: resultados/log_dados_relatorio.txt)

import numpy as np
from scipy.stats import binomtest, chi2

import cv_comparacao as cv

N_SORTEIOS = 2000


def mcnemar(a, b):
    """a, b: acertos (bool) dos dois modelos. Devolve b10, b01, qui-quadrado corrigido, p qui-quadrado, p exato."""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    est = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return n10, n01, est, chi2.sf(est, 1), binomtest(n01, n10 + n01, 0.5).pvalue


def main():
    _, y, fold = cv.oof_csv('bertimbau')
    n = len(y)
    P = {k: cv.oof_transformer(k, s, n, fold) for k, s in cv.TRANSFORMERS.items()}
    P['nbsvm'] = cv.oof_csv('nbsvm')[0]
    ep1 = cv.CLASSES[sum(w * cv.oof_csv(k)[0] for k, w in cv.PESOS_EP1.items()).argmax(1)] == y
    e = cv.CLASSES[sum(P[k] for k in cv.VERSAO_E).argmax(1)] == y
    e5 = cv.CLASSES[sum(P[k] for k in cv.VERSAO_E if k != 'xlmr_large_suave').argmax(1)] == y

    print('=== 1. McNemar na validação cruzada (versão final x ensemble do EP1) ===')
    print('cenário | n | acc versão final | acc EP1 | dif. | só EP1 acerta | só versão final acerta | qui-quadrado | p qui-quadrado | p exato')
    todos, f25 = np.ones(n, bool), fold > 1
    for nome, a, m in [('versão final, 5 folds', e, todos), ('versão final, folds 2-5', e, f25),
                       ('versão final sem xlmr_large_suave, 5 folds', e5, todos), ('versão final sem xlmr_large_suave, folds 2-5', e5, f25)]:
        n10, n01, est, p_chi, p_ex = mcnemar(ep1[m], a[m])
        print(f'{nome} | {m.sum()} | {a[m].mean():.4f} | {ep1[m].mean():.4f} | {a[m].mean() - ep1[m].mean():+.4f} | '
              f'{n10} | {n01} | {est:.2f} | {p_chi:.2g} | {p_ex:.2g}')

    print(f'\n=== 2. Poder do McNemar com 900 textos (300 por classe), {N_SORTEIOS} sorteios das previsões OOF da CV ===')
    rng = np.random.default_rng(42)
    por_classe = [np.where(y == c)[0] for c in cv.CLASSES]
    difs, ps = [], []
    for _ in range(N_SORTEIOS):
        idx = np.concatenate([rng.choice(ix, 300, replace=False) for ix in por_classe])
        difs.append(e[idx].mean() - ep1[idx].mean())
        ps.append(mcnemar(ep1[idx], e[idx])[4])
    difs, ps = np.array(difs), np.array(ps)
    print(f'diferença média versão final - EP1: {difs.mean():+.4f} (dp {difs.std():.4f}); '
          f'percentis 2,5% e 97,5%: {np.percentile(difs, 2.5):+.4f} e {np.percentile(difs, 97.5):+.4f}')
    print(f'fração dos sorteios com p < 0,05 e versão final melhor (poder): {np.mean((ps < 0.05) & (difs > 0)):.1%}')
    print(f'fração dos sorteios em que a versão final fica abaixo do EP1: {np.mean(difs < 0):.1%}')


if __name__ == '__main__':
    main()
