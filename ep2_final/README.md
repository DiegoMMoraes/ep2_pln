# EP2 (PLN, grupo 4): versão final

Esta pasta tem só o necessário para reproduzir o modelo final do EP2, a entrega do test2, a comparação com o EP1 e as análises do relatório. Os experimentos exploratórios (busca de hiperparâmetros, stacking, LLMs, aprendizado confiante etc.) ficam em `../ep2/`, com o diário em `../ep2/EXPERIMENTOS.md`. O relatório está em `../RELATORIO_EP2.md`.

**Modelo final:** média simples das probabilidades de 6 componentes, todos treinados com 100% do `train.xlsx`:
- 3 BERTimbau large: sementes 42 e 43, e um com rótulos suaves;
- 2 XLM-RoBERTa large: um com rótulos suaves e outro sem;
- 1 NB-SVM.

| Avaliação | Versão final | Ensemble do EP1 |
| :--- | :--- | :--- |
| Validação cruzada de 5 folds (20.092 textos) | **0,4806** (melhor nos 5 folds; McNemar p < 0,001) | 0,4705 |
| test1 (900 textos) | 0,4889 (McNemar p = 0,59) | 0,4967 |

## Estrutura

```
ep2_final/
├── dados/                      train.xlsx, test1.xlsx (sem rótulos), test2.xlsx,
│                               test1-gabarito.xlsx (gabarito do professor), test1-rotulado-ep1.xlsx (entrega do EP1),
│                               folds_5.csv e divisao_80_20.csv (folds agrupados usados no Kaggle)
├── entrega/
│   ├── test2-rotulado-ep2.xlsx ENTREGA (renomear para test2.xlsx dentro do grupoXX.zip)
│   └── test1-rotulado-ep2.xlsx usado só no teste de McNemar EP1 x EP2
├── kaggle/
│   ├── ep2_transformers.ipynb  notebook do Kaggle: treina os 5 transformers (modo final)
│   ├── ep2_treino.py           script de treino (mesmo código da célula %%writefile do notebook)
│   └── kernel-metadata.json    metadados para o envio pelo CLI do Kaggle
├── src/                        data_loader, deduplicacao (chave_texto), divisoes (folds), nbsvm
├── saidas/
│   ├── transformers_final/     probabilidades dos 5 transformers no test1 e no test2 (+ log, histórico, config)
│   ├── transformers_cv/        probabilidades out-of-fold dos folds 2 a 5 (Kaggle, modo cv)
│   ├── transformers_holdout/   probabilidades do fold 1 (holdout 80/20) e índices da avaliação
│   ├── oof/                    OOF de 5 folds do EP1 (bertimbau, svc_word12_char25, logreg_word12) e do NB-SVM
│   ├── nbsvm/                  probabilidades do NB-SVM final no test1 e no test2
│   └── ensemble/               probabilidades da versão final no test1 e no test2
├── resultados/                 saídas dos scripts de comparação e de análise
├── nbsvm_final.py              treina o NB-SVM final e prevê test1 e test2
├── ensemble_final.py           combina os 6 componentes e grava a entrega (confere formato)
├── comparar_teste1.py          acurácia, F1, kappa, matriz de confusão e McNemar no test1
├── nbsvm_cv.py                 validação cruzada de 5 folds do NB-SVM
├── cv_comparacao.py            validação cruzada: 3 melhores modelos isolados e versão final x EP1
├── dados_relatorio.py          robustez da CV, qui-quadrado do McNemar e poder estatístico do test1
├── analise_erros.py            análise de erros (ver ANALISE_ERROS.md)
└── analise_caracteristicas.py  interpretação global e local (ver ANALISE_CARACTERISTICAS.md)
```

## Requisitos

Python 3.10 ou superior. Versões usadas: Python 3.13.14, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1.

```bash
pip install pandas openpyxl numpy scipy scikit-learn joblib statsmodels lime torch transformers
```

No Kaggle (GPU T4 x2) foram usados torch 2.11.0+cu130 e transformers 5.20.2, que já vêm instalados.

## Como rodar

Execute os comandos **de dentro desta pasta**, porque os caminhos são relativos a ela.

```bash
cd ep2_final
```

### 1. Gerar a entrega a partir das saídas já salvas (sem GPU)

As probabilidades dos transformers já estão em `saidas/transformers_final/`. Estes dois comandos refazem a entrega e conferem o formato: abas, colunas, número de linhas, textos na mesma ordem e rótulos válidos.

```bash
python nbsvm_final.py
```

```bash
python ensemble_final.py
```

O resultado é idêntico ao `entrega/test2-rotulado-ep2.xlsx` entregue.

### 2. Comparar com o EP1 no test1 (McNemar)

```bash
python comparar_teste1.py --ep2 entrega/test1-rotulado-ep2.xlsx --saida resultados/comparacao_teste1_ep1_ep2.txt
```

### 3. Validação cruzada de 5 folds

```bash
python nbsvm_cv.py
```

```bash
python cv_comparacao.py
```

```bash
python dados_relatorio.py
```

Saídas: `resultados/cv_comparacao.csv` e `resultados/log_dados_relatorio.txt`. O fold 1 dos transformers vem de `saidas/transformers_holdout/` e os folds 2 a 5 de `saidas/transformers_cv/`.

### 4. Análises do relatório

```bash
python analise_erros.py
```

```bash
python analise_caracteristicas.py
```

- `analise_erros.py` preserva a coluna `categoria_manual` de `resultados/analise_erros_amostra.csv`, que foi preenchida à mão.
- `analise_caracteristicas.py` usa o BERTimbau do EP1 (`../ep1/modelos/bertimbau/final_2ep/final/`) para o LIME. Sem o `model.safetensors` desse modelo, que não está no git, rode com `--sem-bert`.

### 5. Retreinar os transformers (Kaggle, GPU)

1. Crie um Dataset no Kaggle com os 5 arquivos `train.xlsx`, `test1.xlsx`, `test2.xlsx`, `folds_5.csv` e `divisao_80_20.csv` de `dados/`. O notebook encontra os arquivos em qualquer subpasta de `/kaggle/input`.
2. Abra `kaggle/ep2_transformers.ipynb` no Kaggle, adicione o Dataset (*Add Input*), escolha o acelerador **GPU T4 x2**, ligue a Internet e rode *Save Version > Save & Run All*. Os 5 experimentos rodam dois de cada vez, e a rodada leva cerca de 1h50.
3. Baixe a pasta `saida/` e copie cada `final_<experimento>/` para `saidas/transformers_final/`. Depois repita os passos 1 e 2.

Pelo CLI do Kaggle: ajuste `id` e `dataset_sources` em `kaggle/kernel-metadata.json` para o seu usuário e rode os comandos abaixo.

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
```

```bash
kaggle kernels output <usuario>/ep2-final -p saida_kaggle
```

**Observações**
- O notebook está configurado com `salvar_modelo=True`, então também grava os pesos de cada modelo (fp16, cerca de 4,5 GB no total) em `saida/<experimento>/modelo_semente<s>/`. Para gravar só as probabilidades, use `salvar_modelo=False` na célula de configuração.
- Os folds 2 a 5 da validação cruzada usam o mesmo notebook com `modos=['cv']` e `folds_cv=[2, 3]` ou `[4, 5]`.
- O treino na GPU não é determinístico: um retreino produz probabilidades ligeiramente diferentes, e alguns rótulos podem mudar.
- Se um modelo colapsar (mais de 85% das previsões numa só classe), o script retreina com a semente + 1000. No treino final, isso aconteceu com o BERTimbau large de semente 42, que usou a semente 1042, e com o XLM-R large, que usou a 2042. Os arquivos mantêm o nome da semente configurada.
