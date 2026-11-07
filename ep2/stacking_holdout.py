# Stacking na divisão 80/20 (fold 1 agrupado): combina as probabilidades, na avaliação, das configurações de
# holdout_busca/ (geradas por busca_holdout.py) e do BERTimbau do EP1 (OOF do fold 1, mesmo papel: modelo
# treinado nos outros 80%).
#
# O meta-modelo é treinado e avaliado dentro dos 20% de avaliação, com validação cruzada estratificada repetida
# (5 folds x 4 repetições), e comparado com a média simples e com o ensemble entregue no EP1 (que não é treinado
# nesses dados). Também mostra a concordância entre os modelos de base: stacking só ajuda se eles errarem em
# textos diferentes.
#
# Uso: python stacking_holdout.py [--modelos bertimbau logreg_C0.25_w12c25 ...]

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

CLASSES = np.array(['c1', 'c234', 'c5'])
PASTA = 'holdout_busca'
PADRAO = ['bertimbau', 'logreg_C0.25_w12c25', 'svc_C0.05_w13c25', 'ridge_a10_w12c25',
          'nbsvm_C0.25_w12_semacento', 'cnb_a0.1_w12_semacento', 'mnb_a0.03_w13']
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}


def carregar_ep1(nome, idx_av):
    df = pd.read_csv(f'oof/oof_{nome}.csv').sort_values('indice').reset_index(drop=True)
    return df[[f'p_{c}' for c in CLASSES]].to_numpy()[idx_av], df['rotulo'].to_numpy()[idx_av]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelos', nargs='+', default=PADRAO)
    p.add_argument('--saida', default='resultados/stacking_holdout.csv')
    args = p.parse_args()
    warnings.filterwarnings('ignore')

    idx_av = np.load(os.path.join(PASTA, '_indices_avaliacao.npy'))
    _, y = carregar_ep1('bertimbau', idx_av)
    probs = {}
    for nome in args.modelos:
        caminho = os.path.join(PASTA, f'{nome}.npy')
        probs[nome] = np.load(caminho).astype(float) if os.path.exists(caminho) else carregar_ep1(nome, idx_av)[0]

    print(f'Avaliação: {len(y)} textos (fold 1 agrupado)\n\n--- Modelos de base ---')
    prev = {n: CLASSES[probs[n].argmax(1)] for n in args.modelos}
    for n in args.modelos:
        print(f'{n:<34} acc {accuracy_score(y, prev[n]):.4f}')

    print('\n--- Concordância entre as predições (fração de textos com o mesmo rótulo) ---')
    conc = pd.DataFrame(index=args.modelos, columns=args.modelos, dtype=float)
    for a, b in itertools.product(args.modelos, repeat=2):
        conc.loc[a, b] = (prev[a] == prev[b]).mean()
    print(conc.round(2).to_string())
    acertos = np.column_stack([prev[n] == y for n in args.modelos])
    print(f'\nOráculo (algum modelo acerta): {acertos.any(1).mean():.4f} | todos acertam: {acertos.all(1).mean():.4f}')

    linhas = []

    def registrar(nome, accs):
        linhas.append({'metodo': nome, 'acc': round(np.mean(accs), 4), 'dp': round(np.std(accs), 4)})
        print(f'{nome:<55} acc {np.mean(accs):.4f} (±{np.std(accs):.4f})', flush=True)

    rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42).split(y, y))

    def por_particao(previsto):
        # mesma estatística das partições do meta-modelo, para comparar em pé de igualdade
        return [accuracy_score(y[av], previsto[av]) for _, av in rskf]

    print('\n--- Combinações (acc média nas partições da CV do meta-modelo) ---')
    ep1 = sum(w * carregar_ep1(n, idx_av)[0] for n, w in PESOS_EP1.items())
    registrar('EP1: ensemble entregue (sem treino aqui)', por_particao(CLASSES[ep1.argmax(1)]))
    registrar('Média simples das probabilidades', por_particao(CLASSES[sum(probs.values()).argmax(1)]))
    sem_bert = [n for n in args.modelos if not n.startswith('bertimbau')]
    registrar('Média simples sem BERT', por_particao(CLASSES[sum(probs[n] for n in sem_bert).argmax(1)]))

    for rotulo, nomes in [('todos', args.modelos), ('sem BERT', sem_bert)]:
        X = np.hstack([np.log(np.clip(probs[n], 1e-6, 1)) for n in nomes])
        meta = {f'LogReg C={C}': (lambda C=C: LogisticRegression(C=C, max_iter=2000)) for C in [0.01, 0.1, 1]}
        meta['HistGradientBoosting'] = lambda: HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, random_state=42)
        for nome_meta, criar in meta.items():
            accs = [accuracy_score(y[av], criar().fit(X[tr], y[tr]).predict(X[av])) for tr, av in rskf]
            registrar(f'Stacking {nome_meta} [{rotulo}]', accs)

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    pd.DataFrame(linhas).to_csv(args.saida, index=False)
    print(f'\nResultados -> {args.saida}')


if __name__ == '__main__':
    main()
