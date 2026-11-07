# Análise de erros da versão E no test1 (aula 08: identificar classes de problemas nas instâncias erradas).
#
# 1. Matriz de confusão e tipos de confusão mais frequentes.
# 2. Taxa de acerto por característica do texto (detectadas automaticamente), ligadas às categorias do slide:
#    - problemas de rotulação: texto repetido do treino com rótulos divergentes; resposta que remete a anexo/link
#      (a informação que o cidadão avaliou não está no texto);
#    - instâncias difíceis: baixa confiança do ensemble; textos muito curtos;
#    - possível erro de rótulo: erro com alta confiança.
# 3. Amostra estratificada de erros (10 por tipo de confusão) em resultados/analise_erros_amostra.csv,
#    com uma coluna para a classificação manual.
#
# Uso: python analise_erros.py   (requer ../test1-rotulado-ep2.xlsx e finais/ensemble/probabilidades_test1.csv)

import re

import numpy as np
import pandas as pd

from src.deduplicacao import chave_texto

CLASSES = ['c1', 'c234', 'c5']
PADROES = {
    'remete a anexo/link': r'\banexo|\banexad|em anexo|segue[m]? (a |o )?(resposta|arquivo|documento|planilha)|https?://|www\.|\blink\b|disponível (no|em) (site|endereço|sítio)',
    'pede esclarecimento/reformulação': r'reformul|esclare[cç]|não está claro|especifi(que|car)|não foi possível identificar|complement',
    'nega/sem informação': r'não (possu|disp[oõ]|dispomos|detém|localizamos|há)|inexist|não compete|não é competência|incompet|sigil|negad|indeferid',
    'encaminha a outro órgão': r'encaminh|redirecion|outro órgão|órgão competente|deverá (ser )?(dirigid|solicitad)',
}


def main():
    gab = pd.read_excel('../test1-rotulado-para-teste-estatistico.xlsx')
    textos = gab['resp_text'].fillna('').astype(str)
    y = gab['clarity'].to_numpy()
    prev = pd.read_excel('../test1-rotulado-ep2.xlsx')['clarity'].to_numpy()
    probs = pd.read_csv('finais/ensemble/probabilidades_test1.csv')[[f'p_{c}' for c in CLASSES]].to_numpy()
    confianca = probs.max(1)
    acerto = prev == y

    treino = pd.read_excel('../train.xlsx')
    treino['k'] = treino['resp_text'].fillna('').astype(str).apply(chave_texto)
    grupos = treino.groupby('k')['clarity'].agg(['size', 'nunique'])
    chaves = textos.apply(chave_texto)
    repetido = chaves.isin(grupos.index).to_numpy()
    divergente = chaves.map(grupos['nunique']).fillna(0).to_numpy() > 1
    palavras = textos.str.split().str.len().to_numpy()

    print(f'Versão E no test1: acurácia {acerto.mean():.4f} ({acerto.sum()}/{len(y)})\n')
    mc = pd.crosstab(pd.Series(y, name='real'), pd.Series(prev, name='previsto'))
    print('Matriz de confusão:\n' + mc.to_string())
    print('\nRecall por classe: ' + ', '.join(f'{c} {np.mean(prev[y == c] == c):.2f}' for c in CLASSES))
    erros = pd.Series([f'{a}→{b}' for a, b in zip(y[~acerto], prev[~acerto])]).value_counts()
    print(f'\nTipos de erro (real→previsto), {len(erros)} tipos, {(~acerto).sum()} erros:')
    print((erros / (~acerto).sum()).round(3).to_string())
    adjacente = sum(v for k, v in erros.items() if 'c234' in k)
    print(f'Erros envolvendo a classe do meio (c234): {adjacente / (~acerto).sum():.0%}; '
          f'erros entre extremos (c1↔c5): {1 - adjacente / (~acerto).sum():.0%}')

    caracteristicas = {
        'repetido do treino': repetido,
        'repetido com rótulos divergentes no treino': divergente,
        'texto curto (< 30 palavras)': palavras < 30,
        'texto longo (> 300 palavras)': palavras > 300,
        'baixa confiança (máx. prob. < 0,45)': confianca < 0.45,
        'alta confiança (máx. prob. ≥ 0,60)': confianca >= 0.60,
    }
    for nome, padrao in PADROES.items():
        caracteristicas[nome] = textos.str.contains(padrao, flags=re.IGNORECASE, regex=True).to_numpy()
    print('\nAcurácia por característica do texto:')
    linhas = []
    for nome, m in caracteristicas.items():
        dist = pd.Series(y[m]).value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        linhas.append({'caracteristica': nome, 'n': int(m.sum()), 'acuracia': round(acerto[m].mean(), 3),
                       'acuracia_resto': round(acerto[~m].mean(), 3),
                       'dist_real': ' / '.join(f'{c} {v:.0%}' for c, v in dist.items())})
    tab = pd.DataFrame(linhas)
    print(tab.to_string(index=False))
    tab.to_csv('resultados/analise_erros_caracteristicas.csv', index=False)

    # amostra estratificada para leitura
    rng = np.random.default_rng(42)
    sel = []
    for tipo in erros.index:
        real, previsto = tipo.split('→')
        cand = np.where((y == real) & (prev == previsto))[0]
        sel.extend(rng.choice(cand, min(10, len(cand)), replace=False))
    amostra = pd.DataFrame({
        'indice_test1': sel, 'real': y[sel], 'previsto': prev[sel],
        'confianca': confianca[sel].round(2), 'palavras': palavras[sel],
        'repetido_treino': repetido[sel], **{n: caracteristicas[n][sel] for n in PADROES},
        'texto': [textos[i][:600] for i in sel], 'categoria_manual': '',
    })
    amostra.to_csv('resultados/analise_erros_amostra.csv', index=False, encoding='utf-8-sig')
    print(f'\nAmostra de {len(amostra)} erros -> resultados/analise_erros_amostra.csv')


if __name__ == '__main__':
    main()
