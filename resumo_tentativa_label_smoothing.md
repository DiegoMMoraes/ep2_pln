# Resumo do EP2: ensemble de BERTimbau com label smoothing + modelos clássicos

Classificação ternária da clareza das respostas do e-SIC (classes `c1`, `c234`, `c5`), disciplina ACH2118 (Introdução ao Processamento de Língua Natural), EACH-USP.

## 1. Contexto

| Item | Valor |
|---|---|
| Tarefa | Classificar `resp_text` em `c1`, `c234` ou `c5` |
| Treino | `train.xlsx`, 20.092 respostas rotuladas (o mesmo do EP1) |
| Teste 1 | 900 instâncias, usado no EP1; no EP2 vem com rótulos, **só para o teste estatístico EP1 × EP2** (McNemar) |
| Teste 2 | 897 instâncias, avaliação oficial do EP2 (rotulado uma única vez, no final) |
| Baseline oficial | TF-IDF + regressão logística: acurácia 0,4541 |
| Modelo do EP1 | Ensemble BERTimbau + LinearSVC + regressão logística: **0,4706 ± 0,0036** na validação cruzada |

O EP1 já tinha esbarrado em dois problemas: um teto de acurácia perto de 0,47 e um forte **ruído de rótulos** (291 textos idênticos com rótulos conflitantes, somando 1.682 linhas). Todo o trabalho do EP2 partiu dessas duas observações.

## 2. O modelo final

O modelo é um **ensemble de probabilidades** com três componentes. Cada componente prevê probabilidades para as 3 classes, e a decisão é a classe de maior probabilidade na média ponderada.

```
                       texto da resposta (resp_text)
                                   |
          +------------------------+------------------------+
          |                        |                        |
 BERTimbau + label smoothing   LinearSVC               Regressão logística
 (média de 2 sementes)         TF-IDF palavra (1,2)    TF-IDF palavra (1,2)
 peso 0,6203                   + caractere (2,5)       peso 0,2748
                               peso 0,1049
          |                        |                        |
          +------------------------+------------------------+
                                   |
                  média ponderada das probabilidades
                                   |
                        classe = argmax  (c1 / c234 / c5)
```

### 2.1 Componente 1: BERTimbau com label smoothing (média de sementes)

- **Base:** `neuralmind/bert-base-portuguese-cased`, com fine-tuning. Camada de classificação de 3 classes acoplada ao token `[CLS]`.
- **Configuração:** 2 épocas, AdamW com lr 2e-5, aquecimento linear de 10% e decaimento linear, lote 8 com acumulação de 2 (lote efetivo 16), weight decay 0,01, *gradient clipping* 1,0, precisão mista (fp16).
- **Truncamento:** limite de 512 tokens. Nos 10,9% de textos mais longos, mantêm-se os 128 primeiros e os 382 últimos tokens, porque o final da resposta costuma indicar se o pedido foi atendido, negado ou redirecionado.
- **Label smoothing 0,1:** a entropia cruzada usa rótulos suavizados (`--label-smoothing 0.1`). Em vez de um alvo "100% na classe rotulada", o modelo aprende com um alvo um pouco mais espalhado, o que reduz a confiança excessiva em rótulos que podem estar errados.
- **Sementes:** o mesmo treino é repetido com as sementes **42 e 43**, e as probabilidades das duas são promediadas. A semente controla a inicialização da camada de classificação e a ordem dos lotes.

### 2.2 Componente 2: LinearSVC sobre TF-IDF de palavras e caracteres

Duas representações TF-IDF concatenadas (palavras em n-gramas de 1 a 2; caracteres em n-gramas de 2 a 5 dentro das palavras), ambas com `sublinear_tf`. Classificador LinearSVC com C = 0,05, com probabilidades calibradas por sigmoide (`CalibratedClassifierCV`, 5 folds).

### 2.3 Componente 3: regressão logística sobre TF-IDF de palavras

TF-IDF de unigramas e bigramas com `sublinear_tf`, regressão logística com C = 2,0 e `max_iter` = 3000.

### 2.4 Combinação

- **Média ponderada das probabilidades**, com pesos no simplex (não negativos, somando 1), escolhidos minimizando a **log-loss** nas predições out-of-fold.
- Pesos finais: **BERT-LS 0,6203 · LinearSVC 0,1049 · regressão logística 0,2748**.
- Não há ajuste de viés por classe. O ajuste favorece o F1-macro, mas a métrica da disciplina é a acurácia (isso já foi avaliado no EP1).
- Os modelos clássicos e os BERTs finais são retreinados com 100% do `train.xlsx` antes de rotular os testes. Os pesos vêm da validação cruzada.

> **Observação sobre o BERT large.** A seção do notebook se chama "Label-smoothing + BERTimbau large", mas o BERTimbau large ainda **não foi incorporado** ao modelo (a "Avaliação 4" está em aberto). O modelo descrito aqui é o ensemble de BERTimbau base com label smoothing, SVC e regressão logística.

## 3. Como o trabalho foi feito (notebook `EP_PLN.ipynb`)

O notebook roda no Google Colab com uma GPU **Tesla T4**, com o Google Drive montado como diretório de trabalho. A T4 é muito mais rápida que a GTX 1660 Super do EP1: cerca de 4 a 5 minutos por época de treino do BERTimbau base.

### 3.1 Estrutura do notebook

| Seção | O que faz |
|---|---|
| Setup | Verifica a GPU, instala pacotes, monta o Drive e entra na pasta do projeto |
| EP1 | Reproduz o fluxo do EP1: treino final do BERTimbau e rotulação com `prever_ensemble.py` |
| Comparação com a entrega | Compara o resultado do notebook com a planilha oficial do EP1 |
| Alternativa 1 | Ensemble de sementes do BERTimbau padrão (`ensemble_sementes.py`) |
| **Alternativa 2** | **Ensemble de grupos de sementes + label smoothing** (`ensemble_grupos.py`): o modelo deste resumo |
| Avaliação 3 | Comparação pareada, mesma semente com e sem label smoothing |
| Avaliação 4 | BERT large (planejado, ainda não executado) |

### 3.2 Protocolo de validação cruzada

- **Validação cruzada agrupada, 5 folds** (`GroupKFold`, embaralhado, semente 42). Textos idênticos (após normalizar espaços) ficam sempre no mesmo fold.
- **Por quê:** o treino tem muitas respostas padronizadas repetidas. Com uma divisão aleatória, cópias do mesmo texto caem no treino e na validação e inflam o resultado. No EP1, o mesmo LinearSVC deu F1-macro 0,4595 com `StratifiedKFold` e 0,4544 com `GroupKFold`.
- **Os mesmos folds para todos os modelos.** O script confere isso e interrompe com erro se os folds, as linhas ou os rótulos divergirem.
- **Probabilidades out-of-fold:** cada texto é previsto por um modelo que não o viu no treino.
- **Cross-fitting dos pesos:** em cada fold, os pesos do ensemble são aprendidos nos outros 4 folds e aplicados ao fold avaliado. A acurácia reportada é a **média das acurácias por fold**, e o ± é o desvio padrão entre folds.
- **Teste1 e teste2 nunca entram na escolha do modelo.** A seleção é feita só na validação cruzada do treino.

### 3.3 Passo a passo executado (Alternativa 2)

**1. Validação cruzada do BERTimbau com label smoothing, uma vez por semente (~45 min por semente na T4):**

```bash
for s in 42 43:
  python finetune_bertimbau.py --modo cv --epocas 2 --label-smoothing 0.1 \
         --semente $s --saida modelos/bertimbau/cv_ls_s$s
```

Cada fold é salvo ao terminar, então uma sessão interrompida retoma do fold seguinte. Os folds são os mesmos do EP1, porque `--semente-divisao` fica em 42.

**2. Avaliação dos grupos e das combinações.** O grupo `bert` reúne as 3 sementes do BERTimbau padrão (42, 43 e 44, a 42 vinda do EP1) e o grupo `bert_ls` as 2 sementes com label smoothing. O script usa os componentes clássicos (`svc_word12_char25` e `logreg_word12`) lidos de `modelos/cv/oof_*.csv`:

```bash
python ensemble_grupos.py avaliar \
  --grupo bert=modelos/bertimbau/cv_2ep,modelos/bertimbau/cv_2ep_s43,modelos/bertimbau/cv_2ep_s44 \
  --grupo bert_ls=modelos/bertimbau/cv_ls_s42,modelos/bertimbau/cv_ls_s43
```

O script mostra cada semente sozinha, a média de cada grupo, a referência do EP1 (primeira pasta do primeiro grupo + clássicos) e **todas as combinações de grupos**, com a diferença de acurácia contra a referência e o número de folds em que cada combinação ganha. Ele grava a receita (componentes e pesos) da melhor combinação em `modelos/receita_grupos.json`.

**3. Treino final com 100% dos dados, uma vez por semente (~10 min por semente):**

```bash
for s in 42 43:
  python finetune_bertimbau.py --modo final --epocas 2 --label-smoothing 0.1 \
         --semente $s --saida modelos/bertimbau/final_ls_s$s
```

**4. Rotulação do teste1:**

```bash
python ensemble_grupos.py prever --teste data/test1.xlsx \
  --grupo bert_ls=modelos/bertimbau/final_ls_s42/final,modelos/bertimbau/final_ls_s43/final
```

O script aplica os dois BERTs finais (e tira a média), treina ou carrega os modelos clássicos, combina tudo com os pesos da receita e grava `modelos/entrega_ep2/test1.xlsx`. Ele também confere o formato da planilha (abas, colunas, número de linhas, textos na mesma ordem, rótulos válidos).

**5. Comparação com o gabarito.** Uma célula do notebook compara a planilha rotulada com os rótulos reais do teste1 e calcula acurácia, F1-macro, relatório por classe e matriz de confusão.

## 4. Resultados na validação cruzada

### 4.1 BERTimbau sozinho (acurácia média por fold)

| Modelo | Acurácia | F1-macro |
|---|---|---|
| BERTimbau padrão, semente 42 (a do EP1) | 0,4601 ± 0,0029 | 0,4540 |
| BERTimbau padrão, semente 43 | 0,4649 ± 0,0053 | 0,4585 |
| BERTimbau padrão, semente 44 | 0,4648 ± 0,0041 | 0,4563 |
| BERTimbau padrão, **média de 3 sementes** | 0,4683 ± 0,0031 | 0,4606 |
| BERTimbau com label smoothing, semente 42 | 0,4615 ± 0,0079 | 0,4552 |
| BERTimbau com label smoothing, semente 43 | 0,4650 ± 0,0061 | 0,4582 |
| BERTimbau com label smoothing, **média de 2 sementes** | 0,4657 ± 0,0063 | 0,4586 |

### 4.2 Ensembles completos (BERT + LinearSVC + regressão logística)

| Combinação | Acurácia | F1-macro | Δ acurácia vs. EP1 | Folds melhores que o EP1 |
|---|---|---|---|---|
| EP1 (BERT semente 42, sem LS) | 0,4706 | 0,4663 | referência | n/a |
| BERT padrão (3 sementes) | 0,4731 | 0,4674 | +0,25 pt | 4/5 |
| BERT padrão + BERT-LS | 0,4713 | 0,4657 | +0,06 pt | 3/5 |
| **BERT-LS (2 sementes)** | **0,4739** | **0,4693** | **+0,33 pt** | 3/5 |

**O melhor ensemble é BERT-LS (2 sementes) + LinearSVC + regressão logística, com acurácia de 0,4739 em validação cruzada agrupada de 5 folds.** Pesos: 0,6203 / 0,1049 / 0,2748.

### 4.3 Comparação pareada: o mesmo BERT com e sem label smoothing

Aqui o ensemble usa uma única semente de BERT, com e sem label smoothing, mais os clássicos:

| Semente | EP1 (sem LS) | Com LS | Δ | Folds melhores |
|---|---|---|---|---|
| 42 | 0,4706 | 0,4685 | −0,21 pt | 1/5 |
| 43 | 0,4684 | 0,4715 | +0,31 pt | 4/5 |

### 4.4 Como interpretar

- O ganho de 0,4739 sobre 0,4706 é de **+0,33 ponto**, e só em 3 dos 5 folds. O desvio entre folds é de cerca de 0,004 a 0,008, então a diferença **não é estatisticamente confiável**.
- As três combinações de ensemble ficaram numa faixa de 0,26 ponto. Na prática, empatam.
- Escolher a melhor combinação olhando os mesmos folds é levemente otimista.
- O ganho do ensemble de sementes aparece principalmente no BERT sozinho (0,4601 → 0,4683 com 3 sementes). No ensemble completo, o SVC e a regressão logística já cobrem parte da variância, e o ganho encolhe.
- A comparação pareada (seção 4.3) mostra que o **efeito isolado do label smoothing é inconsistente**: piora na semente 42 e melhora na 43.

## 5. Resultados no teste1 contra o gabarito

O teste1 foi rotulado pelos dois métodos e comparado com os rótulos reais, sem usá-lo para escolher o modelo.

| Modelo | Acurácia | Acertos | F1-macro |
|---|---|---|---|
| EP1 (planilha entregue) | 0,4967 | 447 / 900 | 0,4937 |
| **EP2 (ensemble BERT-LS + SVC + regressão logística)** | **0,5078** | **457 / 900** | **0,5054** |
| Diferença | +1,11 pt | +10 acertos | +1,17 pt |

**Leitura honesta do resultado:**

- O novo modelo acerta 10 textos a mais, mas o erro-padrão de uma acurácia com n = 900 é de cerca de 1,7 ponto. Cada número isolado tem essa incerteza.
- Para o **teste de McNemar**, o que conta são os pares em que os dois métodos discordam. Se forem da ordem de 90 a 100 (como entre execuções do mesmo método no EP1), um saldo líquido de 10 acertos dá um p da ordem de 0,3, ou seja, sem significância a 5%. Isso é uma estimativa; o valor real deve ser calculado com as duas planilhas.
- Os dois resultados ficam acima dos ~0,47 da validação cruzada porque o teste1 é balanceado (300 exemplos por classe) e parece ser um pouco mais fácil que a média do treino. Não se deve esperar o mesmo patamar no teste2.

## 6. Por que escolhemos assim

### 6.1 Por que um ensemble com modelos clássicos

No EP1, cada componente isolado ficava perto do baseline (acurácias entre 0,452 e 0,460, contra 0,454). A combinação chegou a 0,4706, porque os modelos **erram em textos diferentes**: o BERT captura sentido e contexto, e o TF-IDF captura palavras e templates exatos (nomes de órgãos, negativas, citações de leis). Ensembles são uma forma barata de reduzir o erro quando o gargalo é variância e ruído, e não falta de capacidade do modelo.

### 6.2 Por que ensemble de sementes

A variância do próprio treino mostrou-se grande:

- O BERTimbau retreinado no Colab concordou com o modelo entregue no EP1 em apenas **84,2%** dos rótulos do teste1. No ensemble completo, **91 de 900 rótulos** diferiram.
- As sementes dão acurácias de 0,4601, 0,4649 e 0,4648. **A semente 42, que foi a do EP1, foi a pior das três**, ou seja, o resultado do EP1 tinha uma parcela de sorte de semente.

Como cada semente erra em exemplos ligeiramente diferentes, a **média das probabilidades** cancela parte dos erros que vêm do acaso do treino. Isso rendeu +0,8 ponto no BERT sozinho (3 sementes) e deixa os rótulos do teste2 menos dependentes de qual semente ou GPU "caiu". O custo é baixo: na T4, cada semente leva cerca de 10 minutos no treino final.

### 6.3 Por que label smoothing

- **Motivação:** o EP1 mostrou muito ruído de rótulos (textos idênticos com rótulos conflitantes) e que, a partir da 3ª época, o BERT passa a memorizar esse ruído. O label smoothing é uma técnica padrão para treinar com rótulos ruidosos: ao suavizar o alvo, o modelo aprende menos a "decorar" rótulos individuais e fica menos confiante demais.
- **Custo:** praticamente nulo. O script de fine-tuning já tinha o parâmetro (`--label-smoothing`), e o valor 0,1 é um padrão da literatura.
- **Resultado:** o grupo BERT-LS foi a combinação de maior acurácia na validação cruzada (0,4739), mas a diferença para o BERT padrão é pequena e a comparação pareada não mostra um efeito consistente. O resultado mais defensável é: **o label smoothing não piorou o modelo e a combinação com ele foi a de maior acurácia, mas o ganho é menor que o ruído da avaliação e não deve ser tratado como comprovado.** Parte do ganho vem simplesmente de promediar sementes.

### 6.4 Por que esse protocolo de avaliação

- **Grupos por texto idêntico** para não vazar cópias do treino na validação.
- **Mesmos folds para todos os modelos** para que as comparações sejam justas.
- **Cross-fitting dos pesos** para que nenhum parâmetro seja escolhido olhando os dados em que é medido.
- **Teste1 apenas para comparar EP1 × EP2** e teste2 uma única vez, para não contaminar a seleção.
- **Pesos por log-loss, avaliação por acurácia:** a log-loss é uma função suave, mais estável para ajustar pesos. A acurácia é a métrica que o EP avalia.

## 7. Limitações e próximos passos

- Os ganhos medidos (+0,3 ponto na validação cruzada, +1,1 ponto no teste1) estão dentro do ruído de avaliação.
- **BERT large** (`neuralmind/bert-large-portuguese-cased`): previsto na Avaliação 4, ainda não executado. Teria de ser testado antes em um fold só, para ver tempo, memória e estabilidade.
- **Componentes de outra natureza**, que aumentam a diversidade do ensemble mais do que sementes do mesmo modelo: embeddings de sentenças (por exemplo BGE-M3 ou multilingual-e5) com um classificador leve.
- **Tratamento explícito do ruído de rótulos:** reduzir o peso ou remover exemplos de treino cujo rótulo discorda fortemente das probabilidades out-of-fold (*confident learning*). Requer cuidado para não vazar informação da validação.
- **Stacking** (meta-classificador sobre as probabilidades dos componentes) em vez da média ponderada.
- **Análise de características** (5% da nota), ainda por fazer: coeficientes mais fortes da regressão logística por classe, análise de erros por tamanho do texto e a matriz de confusão (o `c234` é a classe mais difícil).
- **Teste de McNemar** entre as planilhas do EP1 e do EP2 no teste1, a reportar no relatório.

## 8. Arquivos do projeto

| Arquivo | Função |
|---|---|
| `finetune_bertimbau.py` | Fine-tuning do BERTimbau (modos `holdout`, `cv`, `final`, `prever`), com `--label-smoothing` e `--semente` |
| `ensemble_grupos.py` | Avalia grupos de sementes e combinações com os clássicos por validação cruzada, grava a receita e rotula planilhas |
| `ensemble_sementes.py` | Versão anterior (um único grupo de sementes), usada na Alternativa 1 |
| `prever_ensemble.py` | Rotulação do EP1 e funções de inferência reaproveitadas (`probabilidades_bert`, `probabilidades_classico`) |
| `modelos/receita_grupos.json` | Componentes e pesos do ensemble escolhido |
| `modelos/cv/oof_*.csv` | Probabilidades out-of-fold dos modelos clássicos |
| `modelos/bertimbau/cv_*/cv_folds/` | Probabilidades out-of-fold de cada semente do BERT |
| `modelos/bertimbau/final_ls_s*/final/` | Modelos BERT finais, um por semente |
| `modelos/entrega_ep2/` | Planilhas rotuladas e probabilidades de cada componente |
