# Compara os rótulos do teste1 previstos pelo grupo com o gabarito divulgado pelo professor
# (test1-rotulado-para-teste-estatistico.xlsx).
#
#   - Para cada arquivo de predição: acurácia, F1-macro, kappa, relatório por classe e matriz de confusão.
#   - Com --ep2: teste de McNemar entre EP1 e EP2 (o que o enunciado pede no relatório), usando só as
#     900 instâncias do teste1. Nunca usar o teste2 para comparar modelos.
#
# Uso:
#   python comparar_teste1.py                                   # avalia só o EP1 entregue
#   python comparar_teste1.py --ep2 <teste1 rotulado pelo EP2>.xlsx
#   python comparar_teste1.py --ep2 <arquivo>.xlsx --saida comparacao_teste1.txt

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

CLASSES = ['c1', 'c234', 'c5']
GABARITO = 'test1-rotulado-para-teste-estatistico.xlsx'
EP1 = 'test1-rotulado-ep1.xlsx'


def ler(caminho, referencia=None):
    """Lê a planilha e confere se tem as mesmas linhas (textos, na mesma ordem) que a referência."""
    df = pd.read_excel(caminho, sheet_name=0)
    assert list(df.columns) == ['resp_text', 'clarity'], f'{caminho}: colunas inesperadas {list(df.columns)}'
    assert df['clarity'].isin(CLASSES).all(), f'{caminho}: rótulo ausente ou fora de {CLASSES}'
    if referencia is not None:
        assert len(df) == len(referencia), f'{caminho}: {len(df)} linhas, esperado {len(referencia)}'
        textos = df['resp_text'].fillna('').astype(str)
        assert textos.equals(referencia['resp_text'].fillna('').astype(str)), \
            f'{caminho}: textos diferentes ou fora de ordem em relação ao gabarito'
    return df


def avaliar(nome, gabarito, predito):
    print(f'\n=== {nome} ===')
    print(f'Acurácia: {accuracy_score(gabarito, predito):.4f}')
    print(f'F1-macro: {f1_score(gabarito, predito, average="macro"):.4f}')
    print(f'Kappa:    {cohen_kappa_score(gabarito, predito):.4f}')
    print('\nDistribuição prevista: ' + str(pd.Series(predito).value_counts().reindex(CLASSES, fill_value=0).to_dict()))
    print('\n' + classification_report(gabarito, predito, labels=CLASSES, digits=4, zero_division=0))
    mc = pd.DataFrame(confusion_matrix(gabarito, predito, labels=CLASSES),
                      index=[f'real {c}' for c in CLASSES], columns=[f'prev {c}' for c in CLASSES])
    print('Matriz de confusão:\n' + mc.to_string())


def mcnemar(gabarito, pred_a, pred_b, nome_a='EP1', nome_b='EP2'):
    acerto_a = np.asarray(pred_a) == np.asarray(gabarito)
    acerto_b = np.asarray(pred_b) == np.asarray(gabarito)
    n11 = int(np.sum(acerto_a & acerto_b))     # ambos acertam
    n10 = int(np.sum(acerto_a & ~acerto_b))    # só A acerta
    n01 = int(np.sum(~acerto_a & acerto_b))    # só B acerta
    n00 = int(np.sum(~acerto_a & ~acerto_b))   # ambos erram
    discordantes = n10 + n01

    print(f'\n=== Teste de McNemar: {nome_a} x {nome_b} (n = {len(gabarito)}) ===')
    tabela = pd.DataFrame([[n11, n10], [n01, n00]],
                          index=[f'{nome_a} acerta', f'{nome_a} erra'],
                          columns=[f'{nome_b} acerta', f'{nome_b} erra'])
    print(tabela.to_string())
    print(f'\nAcurácia {nome_a}: {acerto_a.mean():.4f} | Acurácia {nome_b}: {acerto_b.mean():.4f} '
          f'| diferença ({nome_b} - {nome_a}): {acerto_b.mean() - acerto_a.mean():+.4f}')
    print(f'Pares discordantes: b = {n10} (só {nome_a} acerta), c = {n01} (só {nome_b} acerta)')

    if discordantes == 0:
        print('Sem pares discordantes: os modelos acertam e erram nas mesmas instâncias (p = 1).')
        return
    # Exato: sob H0, b ~ Binomial(b + c, 0.5)
    p_exato = binomtest(n10, discordantes, 0.5, alternative='two-sided').pvalue
    # Qui-quadrado com correção de continuidade de Edwards (1 g.l.)
    estat = (abs(n10 - n01) - 1) ** 2 / discordantes
    p_chi2 = chi2.sf(estat, df=1)
    # Unilateral (H1: B acerta mais que A), útil para o critério "EP2 significativamente superior ao EP1"
    p_unilateral = binomtest(n01, discordantes, 0.5, alternative='greater').pvalue

    print(f'Qui-quadrado (com correção de continuidade) = {estat:.4f}, p = {p_chi2:.4g}')
    print(f'Teste exato (binomial), bilateral:            p = {p_exato:.4g}')
    print(f'Teste exato (binomial), unilateral {nome_b} > {nome_a}: p = {p_unilateral:.4g}')
    if p_exato < 0.05:
        melhor = nome_b if n01 > n10 else nome_a
        print(f'Conclusão: diferença significativa a p < 0,05 ({melhor} é superior).')
    else:
        print('Conclusão: diferença não significativa a p < 0,05.')


class Tee:
    def __init__(self, *saidas):
        self.saidas = saidas

    def write(self, texto):
        for s in self.saidas:
            s.write(texto)

    def flush(self):
        for s in self.saidas:
            s.flush()


def main():
    p = argparse.ArgumentParser(description='Compara o teste1 rotulado pelo grupo com o gabarito do professor')
    p.add_argument('--gabarito', default=GABARITO)
    p.add_argument('--ep1', default=EP1, help='teste1 rotulado entregue no EP1')
    p.add_argument('--ep2', default=None, help='teste1 rotulado pelo modelo do EP2 (ativa o teste de McNemar)')
    p.add_argument('--saida', default=None, help='também grava o relatório neste arquivo .txt')
    args = p.parse_args()

    if args.saida:
        arquivo = open(args.saida, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.__stdout__, arquivo)

    gabarito = ler(args.gabarito)
    print(f'Gabarito: {args.gabarito} ({len(gabarito)} instâncias) '
          + str(gabarito['clarity'].value_counts().reindex(CLASSES).to_dict()))
    print(f'Baseline classe majoritária: acurácia {gabarito["clarity"].value_counts(normalize=True).max():.4f}')

    ep1 = ler(args.ep1, gabarito)
    avaliar(f'EP1 ({args.ep1})', gabarito['clarity'], ep1['clarity'])

    if args.ep2:
        ep2 = ler(args.ep2, gabarito)
        avaliar(f'EP2 ({args.ep2})', gabarito['clarity'], ep2['clarity'])
        mcnemar(gabarito['clarity'], ep1['clarity'], ep2['clarity'])
        concordancia = (ep1['clarity'].values == ep2['clarity'].values).mean()
        print(f'\nConcordância entre os rótulos do EP1 e do EP2: {concordancia:.4f}')


if __name__ == '__main__':
    main()
