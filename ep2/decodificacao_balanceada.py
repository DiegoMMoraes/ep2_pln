# Decodificação balanceada (ajuste a uma distribuição de classes conhecida no teste).
#
# Os conjuntos de teste parecem ter classes balanceadas por construção (900 = 3 x 300 e 897 = 3 x 299), enquanto o
# treino não é balanceado e os modelos preveem c5 demais e c234 de menos. Em vez do argmax, a decodificação
# balanceada escolhe a atribuição de rótulos que maximiza a soma das log-probabilidades sujeita a ~n/3 textos por
# classe (problema de atribuição, resolvido exatamente com linear_sum_assignment).
#
# Validação sem usar o test1: amostras balanceadas do holdout 80/20 (mesmo nº de textos por classe), comparando
# argmax x balanceada para o ensemble do EP1 e para a versão E do EP2.
#
# Uso: python decodificacao_balanceada.py

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

CLASSES = np.array(['c1', 'c234', 'c5'])
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}
VERSAO_E = ['bertimbau_large_suave', 'bertimbau_large', 'bertimbau_large_s43', 'xlmr_large', 'xlmr_large_suave',
            'nbsvm_C0.25_w12_semacento']


def decodificar_balanceado(probs, proporcoes=None):
    """Rótulos que maximizam sum(log p) com a contagem de cada classe fixada (por padrão, n/3 cada)."""
    n, k = probs.shape
    proporcoes = np.full(k, 1 / k) if proporcoes is None else np.asarray(proporcoes)
    cotas = np.floor(proporcoes * n).astype(int)
    cotas[: n - cotas.sum()] += 1                      # distribui o resto
    colunas = np.repeat(np.arange(k), cotas)           # uma coluna por "vaga" de cada classe
    custo = -np.log(np.clip(probs, 1e-9, 1))[:, colunas]
    linhas, escolhidas = linear_sum_assignment(custo)
    rotulos = np.empty(n, dtype=int)
    rotulos[linhas] = colunas[escolhidas]
    return rotulos


def oof_ep1(nome, idx):
    d = pd.read_csv(f'oof/oof_{nome}.csv').sort_values('indice')
    return d[[f'p_{c}' for c in CLASSES]].to_numpy()[idx], d['rotulo'].to_numpy()[idx]


def main():
    idx = np.load('holdout_busca/_indices_avaliacao.npy')
    y = oof_ep1('bertimbau', idx)[1]
    p_ep1 = sum(w * oof_ep1(n, idx)[0] for n, w in PESOS_EP1.items())
    p_e = sum(oof_ep1(n, idx)[0] if n == 'bertimbau' else np.load(f'holdout_busca/{n}.npy').astype(float)
              for n in VERSAO_E) / len(VERSAO_E)
    print('Holdout completo (não balanceado):', pd.Series(y).value_counts().reindex(CLASSES).to_dict())

    rng = np.random.default_rng(42)
    por_classe = {c: np.where(y == c)[0] for c in CLASSES}
    m = min(len(v) for v in por_classe.values())
    tamanho = 300   # como o test1: 300 por classe
    res = {k: [] for k in ['EP1 argmax', 'EP1 balanceada', 'E argmax', 'E balanceada']}
    pares = []
    for _ in range(200):
        amostra = np.concatenate([rng.choice(por_classe[c], tamanho, replace=False) for c in CLASSES])
        ya = y[amostra]
        acertos = {
            'EP1 argmax': CLASSES[p_ep1[amostra].argmax(1)] == ya,
            'EP1 balanceada': CLASSES[decodificar_balanceado(p_ep1[amostra])] == ya,
            'E argmax': CLASSES[p_e[amostra].argmax(1)] == ya,
            'E balanceada': CLASSES[decodificar_balanceado(p_e[amostra])] == ya,
        }
        for k, v in acertos.items():
            res[k].append(v.mean())
        a, b = acertos['EP1 argmax'], acertos['E balanceada']
        pares.append(binomtest(int((~a & b).sum()), int((a ^ b).sum()), 0.5, alternative='greater').pvalue)

    print(f'\n200 amostras balanceadas de {3 * tamanho} textos do holdout (como o test1):')
    for k, v in res.items():
        print(f'  {k:<16} acc média {np.mean(v):.4f} (dp {np.std(v):.4f})')
    print(f'\nGanho médio da balanceada sobre o argmax: EP1 {np.mean(res["EP1 balanceada"]) - np.mean(res["EP1 argmax"]):+.4f}'
          f' | E {np.mean(res["E balanceada"]) - np.mean(res["E argmax"]):+.4f}')
    print(f'E balanceada vs EP1 entregue (argmax): ganho médio {np.mean(res["E balanceada"]) - np.mean(res["EP1 argmax"]):+.4f};'
          f' McNemar unilateral p < 0,05 em {np.mean(np.array(pares) < 0.05):.0%} das amostras (mediana p = {np.median(pares):.3f})')


if __name__ == '__main__':
    main()
