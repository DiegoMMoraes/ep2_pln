# EP1 (PLN-grupo-4): como rodar

Esta pasta tem só o necessário para rotular arquivos com os dois modelos finais do EP1, e os resultados no teste1.
O repositório completo (experimentos, validação cruzada, relatório) fica em `Downloads/PLN/PLN-grupo-4`.

## Requisitos

Python 3.10+ com:

```bash
pip install pandas openpyxl numpy scipy scikit-learn joblib torch transformers
```

A GPU é opcional: o BERTimbau roda na CPU se não houver CUDA, só fica mais lento.

## Como rodar

Execute os comandos **de dentro desta pasta** (`ep1/`), porque os caminhos são relativos a ela.

```bash
cd ep1
```

Modelo entregue no EP1 (ensemble com BERTimbau, sem ajuste de viés):

```bash
python prever_ensemble.py --modelo bert --sem-vies --teste data/test1.xlsx
```

Modelo só com métodos clássicos:

```bash
python prever_ensemble.py --modelo outros_metodos --teste data/test1.xlsx
```

- `--teste` aceita qualquer `.xlsx` com a coluna `resp_text`, por exemplo `--teste ../test2.xlsx`.
- A saída vai para `modelos/entrega/<modelo>/` (mude com `--saida <pasta>`). O arquivo rotulado tem o mesmo nome, a mesma aba e as mesmas linhas que o de entrada, com a coluna `clarity` preenchida. Também são gravadas as probabilidades de cada componente.
- Atenção: rodar com o `data/test1.xlsx` sobrescreve os resultados guardados em `modelos/entrega/`. Eles serão idênticos, porque os modelos são determinísticos.

## Modelos

As receitas ficam em `modelos/melhores_modelos.json`.

| `--modelo` | Combinação (média ponderada das probabilidades) | Arquivos usados |
| :--- | :--- | :--- |
| `bert` | BERTimbau 0,4787 + SVC palavra+caractere 0,2978 + LogReg 0,2235 | `modelos/bertimbau/final_2ep/final/`, `modelos/finais/svc_word12_char25.joblib`, `modelos/finais/logreg_word12.joblib` |
| `outros_metodos` | SVC palavra+caractere 0,9422 + MultinomialNB 0,0578, com viés c1 −0,10 / c5 −0,15 | `modelos/finais/svc_word12_char25.joblib`, `modelos/finais/multinomialnb_word12.joblib` |

Todos foram treinados com 100% do `data/train.xlsx`. **No repositório git** ficam de fora, por passarem de 100 MB, o `modelos/bertimbau/final_2ep/final/model.safetensors` e o `modelos/finais/svc_word12_char25.joblib`; gere-os com os comandos abaixo. Se faltar algum `.joblib` em `modelos/finais/`, ele é retreinado automaticamente com esse arquivo na primeira execução. O BERTimbau final pode ser retreinado com:

```bash
python finetune_bertimbau.py --modo final --epocas 2 --saida modelos/bertimbau/final_2ep
```

## Resultados no teste1

Resultados medidos contra o gabarito do professor (`../test1-rotulado-para-teste-estatistico.xlsx`, 300 instâncias por classe; baseline de classe majoritária 0,3333):

| Modelo | Arquivo | Acurácia | F1-macro |
| :--- | :--- | :--- | :--- |
| `bert --sem-vies` (**entregue no EP1**) | `modelos/entrega/bert_sem_vies/test1.xlsx` | 0,4967 | 0,4937 |
| `outros_metodos` | `modelos/entrega/outros_metodos/test1.xlsx` | 0,4889 | 0,4895 |

O arquivo entregue também está copiado em `../test1-rotulado-ep1.xlsx`, que é o padrão do script de comparação.

## Comparação EP1 × EP2 (teste de McNemar)

Rode a partir da pasta `EP2/`:

```bash
python comparar_teste1.py
python comparar_teste1.py --ep2 <teste1 rotulado pelo EP2>.xlsx --saida comparacao_teste1.txt
```

## Conteúdo

```text
ep1/
├── prever_ensemble.py        # rotula um .xlsx com o modelo escolhido
├── finetune_bertimbau.py     # treino do BERTimbau e funções de inferência usadas pelo prever_ensemble.py
├── src/                      # leitura dos dados, modelos clássicos, divisões, deduplicação
├── data/                     # train.xlsx e test1.xlsx (sem rótulos) do EP1
└── modelos/
    ├── melhores_modelos.json # receitas dos dois ensembles (componentes, pesos, viés)
    ├── bertimbau/final_2ep/final/  # BERTimbau ajustado (2 épocas)
    ├── finais/               # SVC palavra+caractere, LogReg e MultinomialNB treinados
    └── entrega/              # resultados no teste1 (rótulos + probabilidades por componente)
```
