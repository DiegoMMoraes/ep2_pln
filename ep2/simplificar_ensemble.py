# Simplificação do ensemble: avalia a média simples de todos os subconjuntos dos componentes na divisão 80/20.
#
# Escolher o melhor subconjunto e medir nos mesmos textos superestima a acurácia. Por isso, para cada tamanho k,
# além da melhor acurácia "aparente", calcula uma estimativa sem esse viés: sorteia metade dos textos, escolhe o
# melhor subconjunto de tamanho k nessa metade e mede na outra metade (50 repetições).
# Também mostra quantas vezes cada componente entra nos melhores subconjuntos e o custo de treino no Kaggle.
#
# Uso: python simplificar_ensemble.py

import itertools
import os

import numpy as np
import pandas as pd

CLASSES = np.array(['c1', 'c234', 'c5'])
COMPONENTES = ['bertimbau_large_suave', 'bertimbau_large', 'bertimbau_large_s43', 'xlmr_large', 'xlmr_large_suave',
               'mmbert_base', 'qwen3_1.7b', 'bertimbau', 'logreg_C0.25_w12c25', 'nbsvm_C0.25_w12_semacento']
# minutos de GPU no Kaggle para o modelo final (100% do treino, 2 épocas); 0 = roda localmente ou já existe
CUSTO_MIN = {'bertimbau_large_suave': 30, 'bertimbau_large': 30, 'bertimbau_large_s43': 30, 'xlmr_large': 35,
             'xlmr_large_suave': 35, 'mmbert_base': 20, 'qwen3_1.7b': 175, 'bertimbau': 0,
             'logreg_C0.25_w12c25': 0, 'nbsvm_C0.25_w12_semacento': 0}
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}


def oof_ep1(nome, idx):
    d = pd.read_csv(f'oof/oof_{nome}.csv').sort_values('indice')
    return d[[f'p_{c}' for c in CLASSES]].to_numpy()[idx], d['rotulo'].to_numpy()[idx]


def main():
    idx = np.load('holdout_busca/_indices_avaliacao.npy')
    y = oof_ep1('bertimbau', idx)[1]
    probs = {n: oof_ep1(n, idx)[0] if n == 'bertimbau' else np.load(f'holdout_busca/{n}.npy').astype(float)
             for n in COMPONENTES}
    acerto_ep1 = CLASSES[sum(w * oof_ep1(n, idx)[0] for n, w in PESOS_EP1.items()).argmax(1)] == y

    # acerto (texto a texto) da média de cada subconjunto
    subconjuntos, acertos = [], []
    for k in range(1, len(COMPONENTES) + 1):
        for sub in itertools.combinations(COMPONENTES, k):
            subconjuntos.append(sub)
            acertos.append(CLASSES[sum(probs[n] for n in sub).argmax(1)] == y)
    acertos = np.array(acertos)                      # (n_subconjuntos, n_textos)
    tamanhos = np.array([len(s) for s in subconjuntos])
    acc = acertos.mean(1)

    # estimativa sem viés de seleção: escolhe numa metade, mede na outra
    rng = np.random.default_rng(42)
    honesta = {k: [] for k in range(1, len(COMPONENTES) + 1)}
    escolhas = {k: [] for k in honesta}
    for _ in range(50):
        metade = rng.permutation(len(y)) < len(y) // 2
        acc_a, acc_b = acertos[:, metade].mean(1), acertos[:, ~metade].mean(1)
        for k in honesta:
            cand = np.where(tamanhos == k)[0]
            melhor = cand[acc_a[cand].argmax()]
            honesta[k].append(acc_b[melhor] - acerto_ep1[~metade].mean())
            escolhas[k].append(subconjuntos[melhor])

    print(f'Ensemble do EP1 nesta divisão: {acerto_ep1.mean():.4f}\n')
    linhas = []
    for k in honesta:
        cand = np.where(tamanhos == k)[0]
        melhor = cand[acc[cand].argmax()]
        sub = subconjuntos[melhor]
        n10 = int((acerto_ep1 & ~acertos[melhor]).sum())
        n01 = int((~acerto_ep1 & acertos[melhor]).sum())
        linhas.append({'k': k, 'melhor_acc_aparente': round(acc[melhor], 4),
                       'ganho_honesto_sobre_EP1': round(np.mean(honesta[k]), 4),
                       'dp_ganho_honesto': round(np.std(honesta[k]), 4),
                       'custo_kaggle_min': sum(CUSTO_MIN[n] for n in sub),
                       'so_EP1_acerta': n10, 'so_novo_acerta': n01, 'componentes': ' + '.join(sub)})
    tabela = pd.DataFrame(linhas)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 200)
    print(tabela.drop(columns='componentes').to_string(index=False))
    print('\nMelhor subconjunto de cada tamanho:')
    for _, l in tabela.iterrows():
        print(f"  k={l['k']:>2}: {l['componentes']}")

    print('\nFrequência de cada componente nos melhores subconjuntos escolhidos nas 50 metades (k = 3 a 6):')
    freq = pd.Series([n for k in range(3, 7) for sub in escolhas[k] for n in sub]).value_counts() / (50 * 4)
    print(freq.reindex(COMPONENTES).fillna(0).sort_values(ascending=False).round(2).to_string())

    os.makedirs('resultados', exist_ok=True)
    tabela.to_csv('resultados/simplificacao_ensemble.csv', index=False)
    pd.DataFrame({'componentes': [' + '.join(s) for s in subconjuntos], 'k': tamanhos, 'acc': acc.round(4)}) \
        .sort_values('acc', ascending=False).to_csv('resultados/simplificacao_todos_subconjuntos.csv', index=False)
    print('\n-> resultados/simplificacao_ensemble.csv, resultados/simplificacao_todos_subconjuntos.csv')


if __name__ == '__main__':
    main()
