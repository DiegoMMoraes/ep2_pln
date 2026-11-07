# Transformers no Kaggle (EP2)

`ep2_transformers.ipynb` roda qualquer modelo da lista `EXPERIMENTOS`, **um por GPU, em paralelo**. Com o acelerador GPU T4 x2, são dois experimentos ao mesmo tempo, e os demais esperam na fila. O notebook aceita:

- **encoders:** BERTimbau, XLM-R, mmBERT;
- **LLMs com LoRA:** o modelo base fica em fp16;
- **LLMs com QLoRA:** o modelo base fica em 4 bits, o que permite rodar modelos de 4 a 8 bilhões de parâmetros numa T4 de 16 GB.

## Arquivos

| Arquivo | Para que serve |
| :--- | :--- |
| `ep2_transformers.ipynb` | notebook para o Kaggle/Colab. É gerado, não edite à mão: rode `python gerar_notebook.py` |
| `ep2_treino.py` | script de treino. O notebook grava uma cópia dele com `%%writefile` |
| `dados/` e `dados_kaggle.zip` | train, test1 (sem rótulos), test2, `divisao_80_20.csv` e `folds_5.csv` (5 folds iguais aos das OOF do EP1) |
| `gerar_notebook.py` | monta o notebook a partir do `ep2_treino.py` |
| `teste_notebook.py` | teste local do notebook, com 300 textos |

## No Kaggle

1. Abra o Dataset já criado (`ep2-pln`), clique em **New Version** e adicione o `folds_5.csv`. Ou crie um Dataset novo com o `dados_kaggle.zip`.
2. Importe o `ep2_transformers.ipynb` e use **Add Input** para escolher o Dataset.
3. Em Settings, escolha **GPU T4 x2** e **Internet On**.
4. Edite a lista `EXPERIMENTOS`. A célula de instruções tem uma tabela de presets.
5. Clique em **Save Version > Save & Run All (Commit)**.
6. Baixe a pasta `saida/`. Cada experimento gera uma subpasta com `log.txt`, `historico.csv` e as probabilidades.

## Trazer para o EP2

Rode a partir de `ep2/`:

```bash
python importar_kaggle.py <saida>/<experimento>/holdout_semente42_epoca2.csv --nome <experimento>
```

```bash
python stacking_holdout.py --modelos <experimento> bertimbau_large bertimbau logreg_C0.25_w12c25 nbsvm_C0.25_w12_semacento
```
