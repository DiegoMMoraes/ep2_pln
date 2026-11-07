# Stacking: um meta-modelo aprende a combinar as probabilidades out-of-fold (OOF) dos modelos de base.
#
# Entrada: oof/oof_<modelo>.csv (colunas indice, fold, p_c1, p_c234, p_c5, rotulo). Todos os arquivos usam os
# mesmos 5 folds agrupados por texto (src/divisoes.folds_cv), então cada linha de cada arquivo foi prevista por
# um modelo que não viu aquele texto.
#
# Avaliação do meta-modelo: validação cruzada sobre as próprias OOF, com os mesmos folds — treina nos 4 folds
# restantes e avalia no fold deixado de fora. Compara com cada modelo isolado e com a média ponderada do EP1.
#
# Uso: python stacking.py [--modelos bertimbau svc_word12_char25 ...] [--saida resultados/stacking.csv]

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ['c1', 'c234', 'c5']
PASTA_OOF = 'oof'
PASTA_OOF_BUSCA = 'oof_busca'
TODOS = ['bertimbau', 'svc_word12', 'svc_word12_char25', 'logreg_word12', 'complementnb_word12',
         'multinomialnb_word12']
# Receita 'bert' do EP1 (entregue sem viés)
PESOS_EP1 = {'bertimbau': 0.4787, 'svc_word12_char25': 0.2978, 'logreg_word12': 0.2235}


def carregar(nomes):
    """Lê oof/oof_<nome>.csv (OOF do EP1) ou, se não existir, oof_busca/<nome>.npy (gerado por busca_classicos.py,
    nas linhas do train.xlsx)."""
    ref = pd.read_csv(os.path.join(PASTA_OOF, 'oof_bertimbau.csv')).sort_values('indice').reset_index(drop=True)
    rotulo, fold = ref['rotulo'].to_numpy(), ref['fold'].to_numpy()
    probs = {}
    for nome in nomes:
        caminho = os.path.join(PASTA_OOF, f'oof_{nome}.csv')
        if os.path.exists(caminho):
            df = pd.read_csv(caminho).sort_values('indice').reset_index(drop=True)
            assert (df['rotulo'].to_numpy() == rotulo).all() and (df['fold'].to_numpy() == fold).all(), \
                f'{nome}: linhas ou folds diferentes dos outros modelos'
            probs[nome] = df[[f'p_{c}' for c in CLASSES]].to_numpy()
        else:
            probs[nome] = np.load(os.path.join(PASTA_OOF_BUSCA, f'{nome}.npy')).astype(float)
            assert probs[nome].shape == (len(rotulo), len(CLASSES)), f'{nome}: formato inesperado'
    return probs, rotulo, fold


def atributos(probs, nomes):
    # log-probabilidades: o meta-modelo linear passa a combinar "evidências", como uma média geométrica ponderada
    return np.hstack([np.log(np.clip(probs[n], 1e-6, 1)) for n in nomes])


def cv_meta(criar, X, y, fold):
    previsto = np.empty(len(y), dtype=object)
    for f in np.unique(fold):
        treino, aval = fold != f, fold == f
        previsto[aval] = criar().fit(X[treino], y[treino]).predict(X[aval])
    accs = [accuracy_score(y[fold == f], previsto[fold == f]) for f in np.unique(fold)]
    return accuracy_score(y, previsto), np.std(accs), f1_score(y, previsto, average='macro')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelos', nargs='+', default=TODOS)
    p.add_argument('--saida', default='resultados/stacking.csv')
    args = p.parse_args()
    warnings.filterwarnings('ignore')

    probs, y, fold = carregar(args.modelos)
    linhas = []

    def registrar(nome, acc, dp, f1):
        linhas.append({'metodo': nome, 'acc': round(acc, 4), 'acc_dp_folds': round(dp, 4), 'f1_macro': round(f1, 4)})
        print(f'{nome:<60} acc {acc:.4f} (±{dp:.4f})  F1 {f1:.4f}', flush=True)

    def registrar_previsto(nome, previsto):
        accs = [accuracy_score(y[fold == f], previsto[fold == f]) for f in np.unique(fold)]
        registrar(nome, accuracy_score(y, previsto), np.std(accs), f1_score(y, previsto, average='macro'))

    print('--- Modelos isolados (OOF) ---')
    for n in args.modelos:
        registrar_previsto(n, np.array(CLASSES)[probs[n].argmax(1)])

    if all(n in probs for n in PESOS_EP1):
        soma = sum(w * probs[n] for n, w in PESOS_EP1.items())
        registrar_previsto('EP1: média ponderada BERT+SVCwc+LogReg (sem viés)', np.array(CLASSES)[soma.argmax(1)])

    print('\n--- Stacking (meta-modelo avaliado por CV sobre as OOF) ---')
    conjuntos = {'todos': args.modelos}
    sem_bert = [n for n in args.modelos if n != 'bertimbau']
    if len(sem_bert) < len(args.modelos):
        conjuntos['sem BERT'] = sem_bert
    for rotulo_conj, nomes in conjuntos.items():
        X = atributos(probs, nomes)
        for C in [0.01, 0.1, 1.0, 10.0]:
            registrar(f'LogReg C={C} [{rotulo_conj}]',
                      *cv_meta(lambda: LogisticRegression(C=C, max_iter=2000), X, y, fold))
        registrar(f'HistGradientBoosting [{rotulo_conj}]',
                  *cv_meta(lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                  max_leaf_nodes=15, l2_regularization=1.0,
                                                                  random_state=42), X, y, fold))

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    pd.DataFrame(linhas).to_csv(args.saida, index=False)
    print(f'\nResultados -> {args.saida}')


if __name__ == '__main__':
    main()
