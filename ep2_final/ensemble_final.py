# Ensemble final do EP2 (versão final): média simples das probabilidades de 6 componentes treinados em 100% do treino.
#   - 5 transformers (Kaggle, notebook ep2-final): saidas/transformers_final/<experimento>/final_<teste>_semente<s>.csv
#   - NB-SVM (local, nbsvm_final.py): saidas/nbsvm/final_<teste>.csv
#
# Saídas:
#   entrega/test2-rotulado-ep2.xlsx  -> ENTREGA: mesmo nome de aba, colunas e ordem de linhas do test2.xlsx (renomeie para
#                                  o nome exigido na entrega, ex. test2.xlsx dentro do grupoXX.zip)
#   entrega/test1-rotulado-ep2.xlsx  -> só para o teste de McNemar EP1 x EP2 (python comparar_teste1.py --ep2 ...)
#   saidas/ensemble/probabilidades_<teste>.csv
#
# Uso: python ensemble_final.py

import os

import numpy as np
import pandas as pd

CLASSES = np.array(['c1', 'c234', 'c5'])
PASTA_KAGGLE = 'saidas/transformers_final'
COMPONENTES = {
    'bertimbau_large_suave': os.path.join(PASTA_KAGGLE, 'final_bertimbau_large_suave', 'final_{t}_semente42.csv'),
    'bertimbau_large': os.path.join(PASTA_KAGGLE, 'final_bertimbau_large', 'final_{t}_semente42.csv'),
    'bertimbau_large_s43': os.path.join(PASTA_KAGGLE, 'final_bertimbau_large_s43', 'final_{t}_semente43.csv'),
    'xlmr_large': os.path.join(PASTA_KAGGLE, 'final_xlmr_large', 'final_{t}_semente42.csv'),
    'xlmr_large_suave': os.path.join(PASTA_KAGGLE, 'final_xlmr_large_suave', 'final_{t}_semente42.csv'),
    'nbsvm': os.path.join('saidas', 'nbsvm', 'final_{t}.csv'),
}
ORIGINAIS = {'test1': 'dados/test1.xlsx', 'test2': 'dados/test2.xlsx'}   # test1 sem rótulos


def main():
    os.makedirs('saidas/ensemble', exist_ok=True)
    for teste, original in ORIGINAIS.items():
        aba = pd.ExcelFile(original).sheet_names[0]
        df = pd.read_excel(original, sheet_name=aba)
        soma = np.zeros((len(df), len(CLASSES)))
        print(f'\n{teste} ({len(df)} textos):')
        for nome, padrao in COMPONENTES.items():
            p = pd.read_csv(padrao.format(t=teste))
            assert (p['indice'].to_numpy() == np.arange(len(df))).all(), f'{nome}: linhas fora de ordem'
            probs = p[[f'p_{c}' for c in CLASSES]].to_numpy()
            assert np.allclose(probs.sum(1), 1, atol=1e-3), f'{nome}: probabilidades inválidas'
            dist = pd.Series(CLASSES[probs.argmax(1)]).value_counts().reindex(CLASSES, fill_value=0)
            assert dist.max() < 0.85 * len(df), f'{nome}: modelo colapsado ({dist.to_dict()})'
            print(f'  {nome:<24} {dist.to_dict()}')
            soma += probs
        probs = soma / len(COMPONENTES)
        pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES]).to_csv(
            f'saidas/ensemble/probabilidades_{teste}.csv', index_label='indice')

        rotulado = df.copy()
        rotulado['clarity'] = CLASSES[probs.argmax(1)]
        destino = f'entrega/{teste}-rotulado-ep2.xlsx'
        rotulado.to_excel(destino, index=False, sheet_name=aba)

        # conferência do arquivo gravado (erros de formato são penalizados na entrega)
        conferido = pd.read_excel(destino, sheet_name=aba)
        assert pd.ExcelFile(destino).sheet_names == pd.ExcelFile(original).sheet_names, 'abas diferentes'
        assert list(conferido.columns) == list(df.columns), 'colunas diferentes'
        assert len(conferido) == len(df), 'número de linhas diferente'
        assert conferido['resp_text'].fillna('').astype(str).equals(df['resp_text'].fillna('').astype(str)), \
            'textos alterados ou fora de ordem'
        assert conferido['clarity'].isin(CLASSES).all(), 'rótulo ausente ou inválido'
        print(f'  ENSEMBLE                 {conferido["clarity"].value_counts().reindex(CLASSES).to_dict()}')
        print(f'  -> {destino} (aba "{aba}", colunas {list(conferido.columns)}, conferido)')


if __name__ == '__main__':
    main()
