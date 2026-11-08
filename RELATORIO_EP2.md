# Relatório EP2: classificação da clareza de respostas do e-SIC

ACH2118 Introdução ao Processamento de Língua Natural, EACH-USP. Grupo 4.

## 1. Integrantes

| Nome | Nº USP |
| :--- | :--- |
| _(preencher)_ | _(preencher)_ |
| _(preencher)_ | _(preencher)_ |
| _(preencher)_ | _(preencher)_ |
| _(preencher)_ | _(preencher)_ |

## 2. Introdução e estratégia

A tarefa é classificar cada resposta do e-SIC (`resp_text`) em uma de três classes de clareza: `c1`, `c234` e `c5`. O treino é o mesmo do EP1: 20.092 textos de `train.xlsx`. A entrega são os 897 rótulos do `test2.xlsx`. O test1 (900 textos, agora com gabarito) serve só para comparar o EP1 com o EP2 pelo teste de McNemar.

**Modelo final.** É um ensemble de 6 classificadores, combinados pela **média simples das probabilidades**, sem pesos e sem meta-modelo. A classe prevista é a de maior probabilidade média.

| # | Componente | Representação textual | Classificador |
| :--- | :--- | :--- | :--- |
| 1 | BERTimbau large (`neuralmind/bert-large-portuguese-cased`) | embeddings contextuais (subpalavras WordPiece) | fine-tuning com camada de classificação de 3 classes |
| 2 | BERTimbau large, outra semente | idem | idem |
| 3 | BERTimbau large com rótulos suaves | idem | idem, com alvo suave (ver seção 3) |
| 4 | XLM-RoBERTa large (`FacebookAI/xlm-roberta-large`) | embeddings contextuais multilíngues (SentencePiece) | fine-tuning |
| 5 | XLM-RoBERTa large com rótulos suaves | idem | idem, com alvo suave |
| 6 | NB-SVM (Wang e Manning, 2012) | n-gramas binários de palavras (1 e 2), sem acentos | regressão logística um-contra-todos sobre atributos escalados pela razão de log-contagem do Naive Bayes |

A ideia central é a **diversidade**. Os componentes têm arquiteturas e representações diferentes e erram em textos diferentes. Na validação cruzada, nenhum deles sozinho supera o EP1, mas a média dos seis supera (seção 8).

**O modelo do EP1** era um ensemble ponderado de BERTimbau base (peso 0,4787), LinearSVC sobre palavras e caracteres (0,2978) e regressão logística sobre palavras (0,2235) (`ep1/modelos/melhores_modelos.json`).

**Outros modelos avaliados.** Também testamos, entre outros:
- modelos clássicos com busca de hiperparâmetros;
- stacking;
- outros encoders (mmBERT, NorBERTo, moBERTo);
- LLMs Qwen3 de 1.7B, 4B e 8B com LoRA e QLoRA;
- estabilização do treino com EMA e LLRD;
- aprendizado confiante (cleanlab);
- decodificação balanceada;
- um ensemble de BERTimbau base com label smoothing e modelos clássicos.

Esse último teve a maior acurácia no test1, mas não foi escolhido (seção 8.3). Todos os resultados estão na seção 7.

## 3. Pré-processamento

**Comum a todos os modelos**
- Valores ausentes viram texto vazio, e tudo é convertido para string (`src/data_loader.py`).
- **Não removemos duplicatas do treino.** Textos que só diferem em espaços e quebras de linha são tratados como o mesmo texto pela função `chave_texto`, que colapsa espaços em branco. Ela é usada para:
  - agrupar as divisões, de modo que cópias do mesmo texto nunca fiquem no treino e na avaliação ao mesmo tempo;
  - calcular os rótulos suaves.

**Modelos clássicos** (NB-SVM e os testados na busca)
- O texto entra sem normalização própria. A vetorização do scikit-learn aplica minúsculas e o `token_pattern` padrão.
- No NB-SVM final: `CountVectorizer(ngram_range=(1, 2), binary=True, strip_accents='unicode')`.

**Transformers** (`kaggle/ep2_treino.py`)
- Espaços normalizados com `chave_texto`; depois, o tokenizador do próprio modelo. Não removemos acentos nem passamos para minúsculas, porque os modelos são *cased*.
- **Truncamento início + fim:** o limite é de 512 tokens. Nos textos mais longos (10,9% do treino), mantêm-se os 128 primeiros e os 382 últimos tokens. O final da resposta costuma dizer se o pedido foi atendido, negado ou encaminhado. A mediana é de 182 tokens por texto (`kaggle/saida_kaggle/final/saida/final_bertimbau_large_suave/log.txt`).
- **Rótulos suaves** (componentes 3 e 5): o alvo de cada texto de treino é a distribuição dos rótulos de todas as cópias desse texto no treino. Por exemplo, um texto que aparece uma vez como c1 e outra como c5 tem alvo [0,5; 0; 0,5]. Textos sem divergência mantêm o alvo *one-hot*. No treino completo, 1.935 linhas pertencem a grupos com rótulos divergentes. A perda é a entropia cruzada com alvo probabilístico.

## 4. Parâmetros avaliados

### 4.1 Modelos clássicos

A busca foi feita na divisão 80/20 (seção 6), com 343 configurações no total (`busca_holdout.py`, `resultados/busca_holdout.csv`).

| Modelo | Parâmetro | Valores |
| :--- | :--- | :--- |
| Regressão logística | C | {0,1; 0,25; 0,5; 1; 2; 4} |
| Regressão logística balanceada | C | {0,5; 1} |
| LinearSVC | C | {0,02; 0,05; 0,1; 0,2; 0,4} |
| Ridge | α | {1; 3; 10} |
| SGD (*modified huber*) | α | {1e-5; 3e-5; 1e-4} |
| ComplementNB | α | {0,1; 0,3; 0,5; 1,0} |
| MultinomialNB | α | {0,03; 0,1; 0,3} |
| NB-SVM | C | {0,05; 0,1; 0,25; 0,5} |
| Representação (todos os modelos) | conjunto de atributos | 7 de palavras e 7 com caracteres, listados abaixo |

**Conjuntos de atributos**
- **De palavras:** TF-IDF com `sublinear_tf`, com estas variantes:
  - n-gramas (1,1), (1,2) e (1,3);
  - sem `sublinear_tf`;
  - `min_df` = 2 e `max_df` = 0,9;
  - sem acentos;
  - binário.
- **Com caracteres:** `char_wb` (2,5) e (3,6), `char` (2,5), e combinações de palavras com caracteres. O ComplementNB, o MultinomialNB e o NB-SVM foram avaliados só nos conjuntos de palavras.

**Meta-modelo do stacking:** regressão logística com C ∈ {0,01; 0,1; 1} e HistGradientBoosting com parâmetros fixos.

### 4.2 Transformers

Não houve busca em grade. Cada modelo foi avaliado com uma configuração padrão, e algumas variações foram testadas uma a uma (`kaggle/ep2_treino.py`; configurações em `kaggle/saida_kaggle/*/saida/*/config_entrada.json`).

| Parâmetro | Valores considerados |
| :--- | :--- |
| Modelo pré-treinado | BERTimbau large; XLM-RoBERTa large; mmBERT base; NorBERTo base e large; moBERTo; Qwen3-1.7B (LoRA); Qwen3-4B (LoRA); Qwen3-8B (QLoRA, 4 bits) |
| Taxa de aprendizado | 1e-5 (XLM-R); 1,5e-5 e 2e-5 (BERTimbau large); 2e-5 (NorBERTo large); 3e-5 (bases); 1e-4 (Qwen3 com LoRA) |
| Épocas | {1, 2, 3} |
| Tamanho máximo (tokens) | {256, 384, 512, 1024} |
| Lote × acumulação | 8 × 2; 4 × 4 |
| Rótulos suaves | {não, sim} |
| EMA dos pesos | {não, 0,999} |
| LLRD (decaimento da lr por camada) | {não, 0,9} |
| Semente | {42, 43} |

Valores fixos: AdamW com *weight decay* 0,01, *warmup* linear de 10% seguido de decaimento linear, *gradient clipping* 1,0, precisão mista fp16, 128 tokens do início no truncamento e LoRA com r = 16 e α = 32.

## 5. Valores ótimos

### 5.1 Melhor configuração de cada modelo clássico

Acurácia na divisão 80/20 (`resultados/busca_holdout.csv`).

| Modelo | Parâmetro | Valor ótimo | Atributos | Acurácia |
| :--- | :--- | :--- | :--- | :--- |
| Regressão logística | C | 0,25 | palavras 1-2 + caracteres 2-5 | 0,4620 |
| LinearSVC | C | 0,05 | palavras 1-3 + caracteres 2-5 | 0,4610 |
| Ridge | α | 10 | palavras 1-2 + caracteres 2-5 | 0,4608 |
| Regressão logística balanceada | C | 0,5 | palavras 1-3 + caracteres 2-5 | 0,4608 |
| **NB-SVM (usado na versão final)** | C | **0,25** | **palavras 1-2, binário, sem acentos** | 0,4600 |
| ComplementNB | α | 0,1 | palavras 1-2, sem acentos | 0,4547 |
| MultinomialNB | α | 0,03 | palavras 1-3 | 0,4540 |
| SGD (*modified huber*) | α | 1e-4 | palavras 1-2, sem acentos | 0,4527 |

O NB-SVM entrou na versão final no lugar da regressão logística, que tinha acurácia um pouco maior. O critério foi o ganho no ensemble, e não a acurácia isolada (seção 7.3).

### 5.2 Componentes da versão final

| Componente | Taxa de aprendizado | Épocas | Lote × acum. | Tokens (início) | Rótulos suaves | Semente |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BERTimbau large | 1,5e-5 | 2 | 8 × 2 | 512 (128) | não | 42 |
| BERTimbau large | 1,5e-5 | 2 | 8 × 2 | 512 (128) | não | 43 |
| BERTimbau large | 1,5e-5 | 2 | 8 × 2 | 512 (128) | sim | 42 |
| XLM-RoBERTa large | 1e-5 | 2 | 8 × 2 | 512 (128) | não | 42 |
| XLM-RoBERTa large | 1e-5 | 2 | 8 × 2 | 512 (128) | sim | 42 |
| NB-SVM | C = 0,25 | n/a | n/a | n/a | n/a | n/a |

**Proteção contra colapso.** Alguns treinos de modelos *large* colapsaram e passaram a prever uma única classe. O script detecta esse caso: se mais de 85% das previsões em 1.000 textos do treino caem numa só classe, o modelo é descartado e retreinado com a semente + 1000 (até 3 tentativas). No treino final, isso aconteceu no BERTimbau large com semente 42, que foi treinado com a semente 1042, e no XLM-R large, que só convergiu com a semente 2042 (`historico.csv`, coluna `semente_usada`).

**EMA e LLRD** não entraram no modelo final. Na validação cruzada, eles não melhoraram o BERTimbau large (seção 7.2).

## 6. Procedimento

- **Divisão agrupada.** As divisões usam `GroupKFold(n_splits=5, shuffle=True, random_state=42)`, com os grupos definidos por `chave_texto`. Assim, textos idênticos nunca ficam de lados opostos. São os mesmos folds da validação cruzada do EP1.
- **Avaliação rápida (holdout 80/20).** O fold 1 serve de avaliação: 16.116 textos de treino e 3.976 de avaliação. A busca de hiperparâmetros, os transformers e a escolha da versão final foram feitos nessa divisão.
- **Validação cruzada de 5 folds.** Depois da escolha, os 3 melhores modelos isolados e a versão final foram avaliados nos 5 folds (20.092 textos, previsões *out-of-fold*), com tamanhos de fold de 3.976, 4.062, 3.864, 4.158 e 4.032. Essa é a comparação usada para decidir entre a versão final e o EP1 (seção 8).
- **Treino final.** Os 6 componentes foram retreinados com 100% do `train.xlsx` e aplicados ao test1 e ao test2.
- **Uso dos conjuntos de teste.** O test2 foi rotulado só no final. O test1 serviu apenas para o McNemar contra o EP1 e para as análises de erros e de características, feitas depois da escolha do modelo. **Nenhum modelo nem hiperparâmetro foi escolhido pelo test1.**
- **Infraestrutura.**
  - Os transformers foram treinados no Kaggle (GPU Tesla T4 x2, torch 2.11.0+cu130, transformers 5.20.2), com um experimento por GPU.
  - O NB-SVM, o ensemble e as análises rodaram localmente: Python 3.13.14, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, cleanlab 2.9.0 e lime, com GPU GTX 1660 SUPER para o LIME.

## 7. Resultados

### 7.1 Acurácia do modelo final

| Avaliação | Versão final | Ensemble do EP1 |
| :--- | :--- | :--- |
| Holdout 80/20 (fold 1, 3.976 textos) | 0,4874 | 0,4711 |
| Validação cruzada de 5 folds (20.092 textos) | **0,4806** | 0,4705 |
| test1 (900 textos) | 0,4889 | **0,4967** |
| test2 (entrega) | a ser divulgada pelo professor | n/a |

Baselines: a classe majoritária dá 0,3333 no test1, que é balanceado (300 textos por classe). O baseline oficial (TF-IDF + regressão logística) dá 0,4541 (`ep2/resumo_tentativa_label_smoothing.md`).

### 7.2 Modelos isolados

Acurácia no holdout 80/20, na época final. A validação cruzada só foi rodada nos modelos indicados.

| Modelo | Holdout | CV 5 folds | Observação |
| :--- | :--- | :--- | :--- |
| BERTimbau base (EP1) | 0,4590 | 0,4600 | componente do EP1 |
| **BERTimbau large + rótulos suaves** | **0,4728** | 0,4678 | melhor isolado no holdout; na CV, empata com o EP1 (p = 0,39) |
| BERTimbau large + EMA (semente 43) | 0,4668 | 0,4648 | |
| BERTimbau large + EMA + LLRD, 3 épocas | 0,4655 | 0,4635 | |
| BERTimbau large (semente 42) | 0,4653 | 0,4643 | |
| BERTimbau large (semente 43) | 0,4653 | 0,4625 | |
| Qwen3-4B (LoRA, 384 tokens) | 0,4645 | n/a | |
| NorBERTo large (1.024 tokens) | 0,4630 | n/a | |
| XLM-RoBERTa large | 0,4628 | 0,4608 | |
| mmBERT base | 0,4620 | n/a | |
| Regressão logística (melhor clássico) | 0,4620 | n/a | |
| LinearSVC | 0,4610 | n/a | |
| Qwen3-1.7B (LoRA) | 0,4608 | n/a | |
| NB-SVM | 0,4600 | 0,4517 | |
| XLM-RoBERTa large + rótulos suaves | 0,4600 | 0,4456 | colapso parcial no fold 2 da CV: nunca previu c1 |
| NorBERTo large (512 tokens) | 0,4598 | n/a | |
| NorBERTo base | 0,4580 | n/a | |
| moBERTo | 0,4535 | n/a | |
| Qwen3-8B (QLoRA, 256 tokens, 1 época) | 0,4467 | n/a | |
| BERTimbau large + rótulos suaves (semente 43) | 0,3443 | n/a | colapsou (só c5), antes da proteção contra colapso |

O ganho dos modelos isolados no holdout não se confirmou na CV. Nenhum deles supera o ensemble do EP1 (0,4705).

### 7.3 Ensembles e outras técnicas (holdout 80/20)

| Estratégia | Holdout | Observação |
| :--- | :--- | :--- |
| **Versão final: 3 BERTimbau large + 2 XLM-R large + NB-SVM** | **0,4874** | **modelo escolhido** |
| Versão final + regressão logística + BERTimbau base (8 componentes) | 0,4857 | sem ganho |
| D: 2 BERTimbau large + 2 XLM-R + NB-SVM | 0,4842 | |
| Versão final + Qwen3-4B + Qwen3-8B | 0,4834 | LLMs não somam |
| Versão final + NorBERTo large 1.024 | 0,4834 | |
| Média simples de 10 modelos | 0,4826 | |
| 9 BERTimbau large + 2 XLM-R + NB-SVM | 0,4826 | mais sementes de BERT saturam |
| Versão final + Qwen3-4B | 0,4816 | |
| C: 2 BERTimbau large + XLM-R + NB-SVM | 0,4801 | |
| Stacking (regressão logística, C = 0,01, sobre 10 modelos) | 0,4774 | perde para a média simples dos mesmos 10 (0,4826) |
| A: BERTimbau large + XLM-R + NB-SVM | 0,4728 | |
| Stacking com HistGradientBoosting (7 modelos) | 0,4546 | |

- **Simplificação.** Avaliamos os 1.023 subconjuntos dos 10 componentes. Para cada tamanho k, a melhor combinação foi escolhida numa metade do holdout e medida na outra. Escolhendo assim, o ganho sobre o EP1 é negativo com até 3 componentes (k ≤ 3) e passa a +1,0 ponto com 6 (`resultados/simplificacao_ensemble.csv`).
- **Aprendizado confiante** (cleanlab). Ele marcou 44,6% do treino como suspeito. Remover, re-rotular ou reduzir o peso desses textos **piorou** os clássicos: a média de regressão logística e NB-SVM caiu de 0,4635 para 0,4479–0,4620 (`resultados/aprendizado_confiante.csv`).
- **Decodificação balanceada.** Impõe 1/3 de textos por classe nas previsões, como no test1. Em 200 amostras balanceadas de 900 textos do holdout, melhorou o EP1 em +0,30 ponto e **piorou a versão final em −0,64 ponto**, então foi descartada (`resultados/log_decodificacao_balanceada.txt`).

### 7.4 Label smoothing com BERTimbau base

Também avaliamos, no Google Colab (GPU T4), uma linha de experimentos com BERTimbau base e label smoothing (`ep2/resumo_tentativa_label_smoothing.md`). Os folds de validação cruzada eram os mesmos do EP1.

**Configuração:**
- BERTimbau base com *label smoothing* 0,1, com a média de 2 sementes (42 e 43);
- LinearSVC sobre TF-IDF de palavras 1-2 e caracteres 2-5, com C = 0,05 e calibração sigmoide;
- regressão logística sobre TF-IDF de palavras 1-2, com C = 2,0;
- média ponderada com pesos 0,6203, 0,1049 e 0,2748, escolhidos por *log-loss* nas previsões *out-of-fold*, com *cross-fitting*.

**Resultados na validação cruzada** (média das acurácias por fold):

| Variante | Acurácia | F1-macro |
| :--- | :--- | :--- |
| BERTimbau base, semente 42 | 0,4601 ± 0,0029 | 0,4540 |
| BERTimbau base, média de 3 sementes | 0,4683 ± 0,0031 | 0,4606 |
| BERTimbau base + LS, média de 2 sementes | 0,4657 ± 0,0063 | 0,4586 |
| Ensemble EP1 (mesma avaliação) | 0,4706 | 0,4663 |
| Ensemble BERT padrão (3 sementes) + SVC + LogReg | 0,4731 | 0,4674 |
| **Ensemble BERT-LS (2 sementes) + SVC + LogReg** | **0,4739** | **0,4693** |

Na comparação pareada com a mesma semente, o efeito isolado do label smoothing foi inconsistente: −0,21 ponto com a semente 42 e +0,31 ponto com a 43.

**No test1**, o ensemble BERT-LS foi o **único modelo do EP2 com acurácia acima do EP1** (seção 8.3).

## 8. Evolução em relação ao EP1

### 8.1 test1 (900 textos)

| Estratégia | Teste | Acurácia | F1-macro | Kappa |
| :--- | :--- | :--- | :--- | :--- |
| Modelo EP1 (ensemble entregue) | test1 | 0,4967 | 0,4937 | 0,2450 |
| **Modelo EP2 (versão final)** | test1 | 0,4889 | 0,4838 | 0,2333 |

**Tabela de contingência do McNemar** (`comparacao_teste1_ep1_ep2.txt`):

| | EP2 acerta | EP2 erra |
| :--- | :--- | :--- |
| **EP1 acerta** | 381 | 66 |
| **EP1 erra** | 59 | 394 |

**No test1, a versão final não foi melhor que o EP1.** Ela acertou 7 textos a menos, o que corresponde a −0,78 ponto. A diferença **não é estatisticamente significativa**: McNemar com correção de continuidade de 0,288, **p = 0,5915**; o teste exato binomial dá p = 0,5917.

### 8.2 Validação cruzada de 5 folds (20.092 textos): critério da decisão

A comparação foi feita nos mesmos folds agrupados da validação cruzada do EP1 (`cv_comparacao.py`, `resultados/cv_comparacao.csv`).

| Modelo | Acurácia | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | vs EP1 | Folds acima do EP1 | McNemar p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ensemble do EP1 | 0,4705 | 0,4711 | 0,4695 | 0,4767 | 0,4634 | 0,4722 | n/a | n/a | n/a |
| BERTimbau large + rótulos suaves | 0,4678 | 0,4728 | 0,4611 | 0,4821 | 0,4584 | 0,4655 | −0,27 | 2/5 | 0,39 |
| BERTimbau large + EMA (s43) | 0,4648 | 0,4668 | 0,4567 | 0,4643 | 0,4642 | 0,4720 | −0,57 | 1/5 | 0,066 |
| BERTimbau large + EMA + LLRD | 0,4635 | 0,4655 | 0,4702 | 0,4677 | 0,4512 | 0,4635 | −0,70 | 1/5 | 0,029 |
| **Versão final** | **0,4806** | **0,4874** | **0,4759** | **0,4920** | **0,4711** | **0,4774** | **+1,01** | **5/5** | **0,0001** |

**A versão final foi melhor que o EP1 em todos os 5 folds.** Nas 20.092 previsões *out-of-fold*, a diferença é **estatisticamente significativa**:
- 1.447 textos só a versão final acerta, contra 1.244 que só o EP1 acerta;
- McNemar com correção de continuidade de 15,16, **p = 9,9 × 10⁻⁵**; o teste exato dá p = 9,8 × 10⁻⁵.

**Testes de robustez** (`dados_relatorio.py`, `resultados/log_dados_relatorio.txt`)

| Cenário | n | Acurácia versão final | Acurácia EP1 | Diferença | McNemar | p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Só os folds 2-5 (exclui o fold 1, usado para escolher a versão final) | 16.116 | 0,4789 | 0,4703 | +0,86 | 8,55 | 0,0035 |
| Sem o XLM-R com rótulos suaves (5 componentes), 5 folds | 20.092 | 0,4790 | 0,4705 | +0,85 | 10,64 | 0,0011 |
| Sem o XLM-R com rótulos suaves, folds 2-5 | 16.116 | 0,4785 | 0,4703 | +0,81 | 7,61 | 0,0058 |

A vantagem continua significativa quando se exclui o fold usado na escolha do modelo. Ela também se mantém sem o componente que colapsou parcialmente no fold 2.

### 8.3 Por que a versão final foi escolhida, apesar do test1

| Modelo | CV 5 folds | Folds acima do EP1 | test1 | McNemar no test1 vs EP1 |
| :--- | :--- | :--- | :--- | :--- |
| Ensemble do EP1 | 0,4705 | n/a | 0,4967 | n/a |
| Ensemble BERT-LS + SVC + LogReg | 0,4739 | 3/5 | **0,5078** | 1,095; p = 0,2955 |
| **Versão final** | **0,4806** | **5/5** | 0,4889 | 0,288; p = 0,5915 |

McNemar do ensemble BERT-LS: `comparacao_teste1_ep1_label_smoothing.txt`, com 42 textos que só ele acerta contra 32 que só o EP1 acerta. A acurácia de CV desse ensemble é a média por fold (`ep2/resumo_tentativa_label_smoothing.md`). A da versão final é a acurácia sobre as 20.092 previsões; a média por fold dá 0,4808.

O ensemble com label smoothing foi o melhor no test1: +1,11 ponto, ou 10 acertos a mais que o EP1. **Decidimos não usá-lo**, por quatro motivos:
1. **No test1, nenhum dos dois é significativamente diferente do EP1.** O BERT-LS tem p = 0,30 e a versão final, p = 0,59. Com 900 textos, as duas diferenças estão dentro do ruído.
2. **O test1 tem pouco poder estatístico.** Sorteamos 2.000 conjuntos de 900 textos balanceados (300 por classe) das previsões *out-of-fold* da validação cruzada. Neles, a vantagem média da versão final sobre o EP1 é de +1,05 ponto. Mesmo assim, o McNemar só dá p < 0,05 a favor da versão final em **10,9%** dos sorteios, e a versão final fica **abaixo** do EP1 em **16,9%** deles (`resultados/log_dados_relatorio.txt`). Ou seja, ficar abaixo do EP1 no test1 é compatível com um modelo melhor.
3. **Na validação cruzada, a versão final é melhor e de forma consistente.** São 0,4806 contra 0,4739 do BERT-LS e 0,4705 do EP1, com vantagem em **5 de 5 folds**. O BERT-LS ganhou do EP1 em 3 de 5. A CV usa 20.092 textos, 22 vezes o test1.
4. **Escolher pelo test1 seria usá-lo para seleção de modelo.** O enunciado reserva o test1 apenas para o teste estatístico.

**Conclusão.** A versão final **não** mostrou superioridade no test1: lá ela foi 0,78 ponto pior, sem significância (p = 0,59). Já na validação cruzada de 5 folds sobre o conjunto de treino, ela foi superior ao EP1 em **todos os folds**, com diferença **estatisticamente significativa** (McNemar 15,16; p < 0,001). Por isso, ela foi o modelo escolhido para rotular o test2.

## 9. Análise

As análises completas estão em `ep2/ANALISE_CARACTERISTICAS.md` e `ep2/ANALISE_ERROS.md`.

### 9.1 Interpretação global

Os transformers não são diretamente interpretáveis. Por isso, usamos como **modelo de apoio** o **NB-SVM**, que é linear e é um dos componentes da própria versão final. O peso de cada n-grama numa classe é o coeficiente da regressão logística multiplicado pela razão de log-contagem do NB. Os pesos foram calculados diretamente dos coeficientes, que é o mesmo cálculo do `eli5.show_weights`, sem usar a biblioteca (`analise_caracteristicas.py`, `resultados/caracteristicas_globais.csv`).

**N-gramas de maior peso por classe** ("docs" é o número de textos do treino que contêm o n-grama):

| # | c1 | docs | c234 | docs | c5 | docs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | sra comunicamos | 8 | transmito documento | 8 | odila | 46 |
| 2 | popular informa | 19 | no 2016 | 8 | arquivo quadro | 7 |
| 3 | progep ufal | 8 | projetos da | 8 | senhora odila | 45 |
| 4 | seja area | 5 | via email | 9 | 2018 anexo | 7 |
| 5 | luis cesar | 57 | questionamento acerca | 6 | senhora thais | 6 |
| 6 | valores distribuidos | 35 | oficio em | 6 | pedido ouvida | 7 |
| 7 | ana gabriela | 4 | adriel | 4 | desejada sem | 5 |
| 8 | supes | 5 | www2 ifmg | 17 | de finalizacao | 7 |
| 9 | compreender sua | 11 | cgfse digef | 4 | cristiane mendes | 14 |
| 10 | responsavel existem | 7 | milene | 5 | encaminhamos no | 10 |

**Tipos de característica entre as 300 de maior peso por classe:**

| Classe | Conteúdo | Nome próprio ou sigla | Número ou data | Contato ou URL | Saudação | Template |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| c1 | 56,3% | 26,7% | 11,7% | 2,0% | 1,3% | 2,0% |
| c234 | 59,7% | 25,7% | 9,3% | 4,3% | 1,0% | 0,0% |
| c5 | 52,0% | 38,3% | 7,3% | 0,7% | 1,7% | 0,0% |

**O que a interpretação global mostra**
- **Características espúrias.** Entre 40% e 48% das características mais fortes não descrevem a clareza da resposta: são nomes de pessoas, siglas de órgãos, números, contatos e saudações. Os n-gramas de conteúdo também são muito específicos: a mediana é de 4 textos por n-grama entre os 300 de maior peso. É o mesmo fenômeno do exemplo dos cadernos da FSP, discutido na aula 08.
- **Efeito do solicitante.** Os 46 textos com o nome "Odila" são todos c5, e os 57 com "Luis Cesar" são 88% c1. Como o rótulo é dado pelo próprio cidadão, ele depende de quem pergunta. O modelo aprende esse padrão pelo nome que aparece na saudação da resposta.
- **Templates pesam pouco** (0% a 2%), graças à divisão agrupada por texto.
- **Fidelidade do modelo de apoio.** O NB-SVM concorda com a versão final em 73,4% dos textos do test1 (kappa 0,601) e em 71,7% na validação cruzada (kappa 0,575). Com o BERTimbau large, a concordância é de só 60,2% (kappa 0,402). Os dois tipos de modelo usam evidências diferentes, e essa diversidade explica o ganho do ensemble.

### 9.2 Interpretação local

Seis textos do test1 foram explicados: os 3 acertos de maior confiança (um por classe) e o erro de maior confiança de 3 categorias da análise de erros. Para cada texto, mostramos:
- a contribuição de cada n-grama no NB-SVM, equivalente ao `eli5.show_predictions`;
- o **LIME** (Ribeiro et al., 2016; 500 perturbações) no NB-SVM e no BERTimbau do EP1.

Usamos o BERTimbau do EP1 como substituto porque os pesos dos transformers da versão final não estavam disponíveis localmente (`resultados/caracteristicas_locais.csv`).

| test1 | Caso | Real → previsto (confiança) | NB-SVM (maiores pesos) | LIME no BERTimbau |
| :--- | :--- | :--- | :--- | :--- |
| 142 | acerto c1 | c1 → c1 (0,85) | luis cesar, sr luis, luis encaminhamos, gerente executiva | informação, 12527, bancária, interesse |
| 836 | acerto c234 | c234 → c234 (0,61) | 2016 abono, 2011 setembro, anexa planilha, vagas oriundas | vagas, anexa, Portaria, convocações |
| 453 | acerto c5 | c5 → c5 (0,88) | parecer solicitado, solicitacao encaminhamos, sic procuradoria, dr marcelo | cópia, Dr, anexo, encaminhamos |
| 402 | erro: conteúdo em anexo ou link | c234 → c5 (0,57) | uploads sites, encaminhamos anexo, wp, regimento do | anexo, Encaminhamos, Estrutura, inovacao |
| 654 | erro: resposta negativa | c1 → c5 (0,59) | por ja, vaga, de biblioteca, conforme art | biblioteca, nomeação, não, Prezada |
| 610 | erro: rótulo incoerente | c1 → c5 (0,66) | epp, epp cnpj, 229, ltda epp | Concessionário, Infraero, anterior, 2017 |

**O que a interpretação local mostra**
- O NB-SVM decide por **nomes e números**: o nome do solicitante no texto 142, o nome do procurador no 453, CNPJ e "EPP" no 610.
- O BERTimbau decide mais por **palavras de conteúdo e de ação**, como "informação", "cópia", "anexo" e "encaminhamos".
- No texto 142, o NB-SVM acerta pelo motivo errado: decide pelo nome do cidadão, e não pela resposta.
- Nos erros, os dois modelos são guiados pelo **formato** da resposta (remissão a anexo, lista de CNPJs). Nada no texto indica a clareza que o cidadão percebeu.

### 9.3 Análise de erros

Analisamos os erros da versão final no test1 (`analise_erros.py`, `resultados/log_analise_erros.txt`, `resultados/analise_erros_amostra.csv`).

**Matriz de confusão** (acurácia 0,4889; nas linhas, a classe real):

| | Previsto c1 | Previsto c234 | Previsto c5 | Recall |
| :--- | :--- | :--- | :--- | :--- |
| c1 | 149 | 85 | 66 | 0,50 |
| c234 | 91 | 105 | 104 | 0,35 |
| c5 | 49 | 65 | 186 | 0,62 |

- **75%** dos 460 erros envolvem a classe do meio (c234), que é a mais difícil.
- **25%** são entre extremos (c1↔c5).
- **A confiança do ensemble é informativa.** Nos textos com probabilidade máxima abaixo de 0,45 (328 textos), a acurácia é de **0,363**. Nos de confiança de pelo menos 0,60 (151 textos), é de **0,755**.
- Os textos do test1 que também aparecem no treino (100) têm acurácia **menor** que os demais: 0,450 contra 0,494. Quando o texto tem rótulos divergentes no treino (69 casos), a acurácia cai para 0,435.

**Leitura manual de 60 erros** (10 de cada tipo de confusão real → previsto, sorteados com semente 42). Cada erro recebeu uma categoria criada a partir da leitura e depois ligada às categorias da aula 08:

| Categoria | n | % | Classe de problema (aula 08) |
| :--- | :--- | :--- | :--- |
| A. Conteúdo fora do texto: a resposta só remete a anexo, link ou documento | 19 | 32% | falta de características |
| B. Negativa ou encaminhamento: o rótulo parece refletir o desfecho, e não a clareza | 20 | 33% | rotulação |
| C. Resposta informativa com rótulo incoerente (modelo confiante, 0,58 em média) | 6 | 10% | rotulação |
| D. Texto padrão idêntico com rótulos divergentes | 3 | 5% | rotulação |
| E. Texto vazio ou truncado | 2 | 3% | instância difícil |
| F. Fronteira ambígua entre classes vizinhas | 10 | 17% | instância difícil |

**Considerações**
- Cerca de **80% dos erros lidos** (categorias A a D) vêm da definição da tarefa e dos rótulos, e não de falha do modelo:
  - o cidadão avaliou a resposta vendo também o anexo e a própria pergunta, e nenhum dos dois está no conjunto de dados;
  - o rótulo parece misturar clareza com satisfação: respostas negativas claras recebem c1, e encaminhamentos recebem c5.
- Só cerca de 20% são instâncias de fato ambíguas, concentradas na fronteira com c234.
- Isso explica o teto de acurácia próximo de 0,48–0,50. Nenhum modelo passou disso: nem os encoders *large*, nem os LLMs de 4B e 8B, nem o aprendizado confiante.
- A característica que faltaria é o texto do pedido do cidadão, que não está disponível.

**Limitações.** A amostra tem só 60 erros e não segue a proporção real de cada tipo de erro. A categorização foi feita por um único avaliador. O modelo de apoio concorda só moderadamente com a versão final (kappa ≈ 0,6), e o LIME foi aplicado a um BERTimbau substituto.

## 10. Repositório e reprodução

**Repositório:** https://github.com/DiegoMMoraes/ep2_pln

Os arquivos de modelo do EP1 com mais de 100 MB (`model.safetensors` do BERTimbau e `svc_word12_char25.joblib`) ficaram fora do git. O `ep1/README.md` explica como regenerá-los.

**Dependências**
- **Local:**

  ```bash
  pip install pandas openpyxl numpy scipy scikit-learn joblib statsmodels lime cleanlab torch transformers
  ```

- **Kaggle:** torch e transformers já vêm instalados; o notebook instala peft e bitsandbytes, usados só pelos LLMs.

**Passo a passo para reproduzir a entrega**
1. **Dados no Kaggle.** Crie um Dataset com os arquivos de `ep2/kaggle/dados/`: `train.xlsx`, `test1.xlsx`, `test2.xlsx`, `divisao_80_20.csv` e `folds_5.csv`.
2. **Transformers (Kaggle).**
   - Abra o notebook `ep2/kaggle/envio_final/ep2_transformers.ipynb`, com acelerador **GPU T4 x2**, Internet ligada e o Dataset como entrada.
   - Rode *Save & Run All*. São os 5 experimentos em modo `final`, e a rodada leva cerca de 1h50.
   - Baixe a pasta `saida/` para `ep2/kaggle/saida_kaggle/final/saida/`.
   - Pelo CLI:

     ```bash
     kaggle kernels push -p ep2/kaggle/envio_final --accelerator NvidiaTeslaT4
     ```

     ```bash
     kaggle kernels output <usuario>/ep2-final -p ep2/kaggle/saida_kaggle/final
     ```

3. **NB-SVM (local):**

   ```bash
   cd ep2 && python nbsvm_final.py
   ```

4. **Ensemble e entrega:** gera `test2-rotulado-ep2.xlsx`, que é a entrega, e `test1-rotulado-ep2.xlsx`, conferindo abas, colunas e número de linhas.

   ```bash
   cd ep2 && python ensemble_final.py
   ```

5. **McNemar no test1:**

   ```bash
   python comparar_teste1.py --ep2 test1-rotulado-ep2.xlsx --saida comparacao_teste1_ep1_ep2.txt
   ```

6. **Validação cruzada e números do relatório** (a partir de `ep2/`):

   ```bash
   python nbsvm_cv.py
   ```

   ```bash
   python cv_comparacao.py
   ```

   ```bash
   python dados_relatorio.py
   ```

   Os folds 2 a 5 dos transformers rodam no Kaggle em modo `cv`.
7. **Análises:**

   ```bash
   python analise_erros.py
   ```

   ```bash
   python analise_caracteristicas.py
   ```

**Observação sobre reprodutibilidade.** O treino de transformers na GPU não é determinístico. Um retreino produz probabilidades ligeiramente diferentes, e alguns rótulos podem mudar. O diário completo dos experimentos está em `ep2/EXPERIMENTOS.md`.
