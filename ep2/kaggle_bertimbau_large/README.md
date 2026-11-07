# BERTimbau large no Kaggle/Colab

> **Substituído por `../kaggle/`.** O notebook genérico de lá roda este e outros modelos, com os dados em `../kaggle/dados/`. Esta pasta fica só pelo resultado do holdout já rodado (`saida/`).

O notebook `bertimbau_large.ipynb` faz o fine-tuning de `neuralmind/bert-large-portuguese-cased` com o mesmo pipeline do BERTimbau base do EP1. A GTX 1660 local (6 GB) não comporta o large com AdamW.

## Kaggle (recomendado)

1. **Datasets > New Dataset:** envie o `dados_kaggle.zip`, ou os 4 arquivos da pasta `dados/` (train.xlsx, test1.xlsx sem rótulos, test2.xlsx, divisao_80_20.csv).
2. **Code > New Notebook > File > Import Notebook:** importe o `bertimbau_large.ipynb`. Depois use **Add Input** para escolher o dataset.
3. Em **Settings** (painel direito):
   - Accelerator = **GPU T4 x2**. O notebook usa uma GPU. Não use a P100, que não é suportada pelo PyTorch atual.
   - Internet = **On**, para baixar o modelo do Hugging Face. Exige conta com telefone verificado.
4. Rode as 2 primeiras células para conferir a GPU e os dados.
5. Clique em **Save Version > Save & Run All (Commit)**. O notebook roda em segundo plano, com limite de 12 h, e você pode fechar o navegador.
6. Ao terminar, abra a versão salva e, em **Output**, baixe a pasta `saida/`.

No Colab: Ambiente de execução > Alterar tipo > GPU T4. Envie os 4 arquivos para `/content` e rode tudo. A saída fica em `/content/saida`. A sessão gratuita cai se ficar ociosa, por isso o Kaggle é mais seguro para treinos longos.

## Modos

A variável `MODOS` fica na 1ª célula de código.

| Modo | O que faz | Saída | Tempo estimado na T4 |
| :--- | :--- | :--- | :--- |
| `['holdout']` (padrão) | treina nos 80%, prevê os 20% da divisão dos clássicos | `holdout_probabilidades_epoca_{1,2}.csv`, `holdout_historico.csv` | ~3 h |
| `['final']` | treina em 100% do treino, prevê test1 e test2 | `final_probabilidades_test1.csv`, `final_probabilidades_test2.csv` | ~4 h |
| `['holdout', 'final']` | os dois em sequência | todos os anteriores | ~7-8 h |

Os tempos são estimativas, e o log mostra os minutos restantes a cada 200 lotes. Se aparecer "CUDA out of memory", troque para `GRADIENT_CHECKPOINTING = True`, ou para `LOTE = 4` com `ACUMULACAO = 4`.

## Trazer os resultados para o EP2

Copie a pasta `saida/` baixada para `ep2/kaggle_bertimbau_large/saida/` e rode a partir de `ep2/`:

```bash
python importar_kaggle.py kaggle_bertimbau_large/saida/holdout_probabilidades_epoca_2.csv --nome bertimbau_large
```

```bash
python stacking_holdout.py --modelos bertimbau_large bertimbau logreg_C0.25_w12c25 nbsvm_C0.25_w12_semacento cnb_a0.1_w12_semacento
```

Referências na mesma divisão: BERTimbau base 0,4590; melhor clássico 0,4620; ensemble entregue no EP1 0,4711.
