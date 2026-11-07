# EP2: diário de experimentos

Referência (EP1, entregue): ensemble BERTimbau + SVC palavra+caractere + LogReg, média ponderada sem viés.
Teste1 (gabarito do professor): acurácia 0,4967. O teste1 **não** é usado para escolher modelos; serve só para o McNemar.

## 1. Stacking sobre as OOF do EP1 (5 folds agrupados), em 03/11

`python stacking.py` grava os resultados em `resultados/stacking.csv`.

| Método | Acurácia CV |
| :--- | :--- |
| Melhor modelo isolado (BERTimbau) | 0,4600 |
| Ensemble do EP1 | 0,4705 |
| Stacking com LogReg C=0,01 (todos) | 0,4717 |
| Stacking com HistGradientBoosting | 0,4673 |
| Stacking sem o BERT | 0,4625 |

## 2. Busca de hiperparâmetros dos clássicos, divisão 80/20, em 03/11

`python busca_holdout.py` grava os resultados em `resultados/busca_holdout.csv` e o log em `resultados/log_busca_holdout.txt`.

- **Divisão:** fold 1 do GroupKFold agrupado (16.116 de treino, 3.976 de avaliação). O erro padrão da acurácia é de ±0,8 ponto.
- **Busca:** 14 conjuntos de atributos × LogReg, LogReg balanceada, LinearSVC, Ridge, SGD modified-huber, ComplementNB, MultinomialNB e NB-SVM. São 342 configurações no total.
- **Referências no mesmo fold:** ensemble do EP1 0,4711; SVC palavra+caractere 0,4610; BERTimbau 0,4590.

| Modelo | Melhor configuração | Acurácia |
| :--- | :--- | :--- |
| LogReg | C=0,25, palavras 1-2 + caracteres 2-5 | 0,4620 |
| LinearSVC | C=0,05, palavras 1-3 + caracteres 2-5 | 0,4610 |
| Ridge | α=10, palavras 1-2 + caracteres 2-5 | 0,4608 |
| NB-SVM | C=0,25, palavras 1-2 sem acento | 0,4600 |
| ComplementNB | α=0,1, palavras 1-2 sem acento | 0,4547 |
| MultinomialNB | α=0,03, palavras 1-3 | 0,4540 |
| SGD modified-huber | α=1e-4, palavras 1-2 sem acento | 0,4527 |

**Conclusões:**
- Todos os modelos clássicos param em ~0,46. Nenhum hiperparâmetro faz diferença acima do ruído.
- Palavras + caracteres e um C baixo (regularização forte) ajudam um pouco.
- Variações do TF-IDF (sem acento, binário, sem sublinear, min_df) não mudam o resultado.

## 3. Stacking na divisão 80/20, em 03/11

`python stacking_holdout.py` grava os resultados em `resultados/stacking_holdout.csv`. O meta-modelo é avaliado por CV 5×4 dentro dos 3.976 textos de avaliação.

- LogReg, SVC e Ridge concordam em 95-97% das predições, ou seja, são praticamente o mesmo modelo. O BERT concorda só ~68% com eles.
- Oráculo: em 69,5% dos textos, algum modelo acerta. Há espaço para combinar, mas o meta-modelo não consegue separar quem acerta quando.

| Método | Acurácia |
| :--- | :--- |
| Ensemble do EP1 (sem treino nesses dados) | 0,4711 |
| Stacking com LogReg C=0,01 (7 modelos, com o BERT) | 0,4701 |
| Média simples (7 modelos) | 0,4638 |
| Stacking sem o BERT | 0,4605 |
| Stacking com HistGradientBoosting | 0,4546 |

**Conclusão:** com os clássicos de base, o stacking não supera o ensemble do EP1. O ganho do ensemble vem do BERT, que é o único modelo diferente. Para melhorar, é preciso adicionar modelos de base de outra natureza.

## 4. BERTimbau large (`neuralmind/bert-large-portuguese-cased`), holdout 80/20, em 03/11

Rodou no Kaggle com o notebook `kaggle_bertimbau_large/bertimbau_large.ipynb`. A saída está em `kaggle_bertimbau_large/saida/`.

- **Configuração:** a mesma do base do EP1 (128 tokens do início e o restante do fim, até 512; fp16; lote efetivo 16; warmup de 10%), com lr 1,5e-5 e 2 épocas.
- **Tempo:** 12 min por época na GPU do Kaggle.
- **Resultado:** época 1 com 0,4474; época 2 com **0,4653**. O base faz 0,4590 no mesmo fold, então o large ganha +0,6 ponto.
- **Concordância:** o large concorda 77% com o base e 59-67% com os clássicos. Com ele, o oráculo sobe de 69,5% para 72,5%.

`python stacking_holdout.py --modelos ...` grava os resultados em `resultados/stacking_holdout_large*.csv`.

| Combinação | Acurácia |
| :--- | :--- |
| **Média simples: large + base + LogReg w12c25 + NB-SVM** | **0,4761** |
| Stacking LogReg C=0,1, com 7 modelos (large, base e 5 clássicos) | 0,4732 |
| Ensemble do EP1 | 0,4711 |
| Média simples dos 7 modelos | 0,4663 |
| Sem nenhum BERT (melhor caso) | 0,4635 |

**Conclusões:**
- O large é o melhor modelo isolado até aqui.
- O ganho sobre o EP1 nessa divisão é de +0,5 ponto, dentro do ruído de ±0,8. Ainda não está confirmado.
- O meta-modelo treinado em só ~3.200 textos não supera a média simples. Para fazer o stacking direito, precisamos das OOF do large nos 5 folds (validação cruzada) para treinar o meta-modelo nos 20 mil textos.

## 5. Próximas ideias (03/11)

**Dados.** São respostas do e-SIC, e o rótulo é a clareza percebida, muito ruidoso. Exemplo: "segue anexada resposta" aparece como c5. 1.935 linhas estão em grupos de textos idênticos com rótulos diferentes. 100 textos do test1 e 106 do test2 também aparecem no treino.

| Ideia | Origem | Como testar |
| :--- | :--- | :--- |
| Modelos mais novos e maiores: XLM-R large, mmBERT (2025), Qwen3 1.7B/4B/8B com LoRA/QLoRA | pedido do grupo | `kaggle/ep2_transformers.ipynb` |
| Rótulos suaves: o alvo é a distribuição dos rótulos de textos repetidos | aula 06, "rótulos divergentes" | `rotulos_suaves=True` |
| Média de várias sementes do melhor transformer | ensemble barato, reduz a variância | `sementes=[42, 43, 44]` |
| Stacking com as OOF dos transformers (5 folds) | pedido do grupo | modo `cv` + `stacking.py` |
| Semi-supervisionado: pseudo-rótulos nos textos do test1 e do test2 | aula 06, "métodos semi-supervisionados" | depois |
| Destilação: treinar um modelo menor nas probabilidades do ensemble | aula 06, "destilação do conhecimento" | depois |
| Análise de características (LogReg/LIWC, eli5) e de erros | aulas 07-08; 5% da nota | para o relatório |
| McNemar EP1 × EP2 no test1 | aula 09 e enunciado | `../comparar_teste1.py` |

## 6. Rodada 2 no Kaggle (T4 x2), holdout 80/20, em 03/11

`kaggle/ep2_transformers.ipynb`, enviado pelo CLI (`kaggle/envio/`). A saída está em `kaggle/saida_kaggle/run2/`. Duração total: 45 min, com 2 experimentos em paralelo.

| Modelo | Época 1 | Época 2 | Tempo por época |
| :--- | :--- | :--- | :--- |
| **BERTimbau large + rótulos suaves** | 0,4522 | **0,4728** | 11 min |
| BERTimbau large (seção 4) | 0,4474 | 0,4653 | 12 min |
| XLM-R large | 0,4256 | 0,4628 | 13,5 min |
| mmBERT base | 0,4457 | 0,4620 | 8 min |
| Qwen3-1.7B (LoRA) | erro: o torchao do Kaggle é antigo para o peft (corrigido no notebook) | | |

Stacking dos 7 modelos (os 5 transformers, LogReg w12c25 e NB-SVM), em `resultados/stacking_holdout_run2.csv`:

| Combinação | Acurácia |
| :--- | :--- |
| **Média simples dos 7 modelos** | **0,4774** |
| Média simples sem os BERTimbau | 0,4759 |
| Stacking LogReg C=0,01 | 0,4730 |
| Ensemble do EP1 | 0,4711 |

O oráculo sobe para 74%. XLM-R e mmBERT concordam só 72-75% com os BERTimbau, então são diversos, mas sozinhos ficam abaixo deles.

**Conclusões:**
- Os rótulos suaves deram +0,75 ponto no BERTimbau large. Falta separar isso da variação entre sementes.
- A média simples segue melhor que o meta-modelo treinado em 3.200 textos.

## 7. Rodada 3 no Kaggle, holdout 80/20, em 03/11 (19h10–21h30)

| Modelo | Época 2 | Observação |
| :--- | :--- | :--- |
| BERTimbau large, semente 43 | 0,4653 | igual à semente 42 (0,4653) |
| BERTimbau large com rótulos suaves, semente 43 | 0,3443 | **colapsou**: previu c5 para todos os textos (instabilidade conhecida do BERT large) |
| XLM-R large com rótulos suaves | 0,4600 | contra 0,4628 sem rótulos suaves: sem ganho |
| Qwen3-1.7B (LoRA) | 0,4608 | 69 min por época; concorda só ~70% com os BERTimbau |

**Rótulos suaves:** o ganho visto com a semente 42 (+0,75 ponto) não se confirmou. No XLM-R não ajudou, e com a semente 43 o modelo colapsou. Não dá para afirmar que eles ajudam.

O treino ganhou uma proteção contra colapso: se mais de 85% das predições caírem numa só classe, o modelo é retreinado com outra semente (`max_tentativas`).

Combinação dos 10 modelos sem o colapsado (BERTimbau large com rótulos suaves s42, BERTimbau large s42 e s43, XLM-R large com e sem rótulos suaves, mmBERT, Qwen3-1.7B, BERTimbau base, LogReg w12c25 e NB-SVM), em `resultados/stacking_holdout_run3.csv`:

| Combinação | Acurácia |
| :--- | :--- |
| **Média simples dos 10 modelos** | **0,4826** |
| Média sem os BERTimbau | 0,4776 |
| Stacking LogReg C=0,01 | 0,4774 |
| Média só dos 8 transformers | 0,4761 |
| Ensemble do EP1 | 0,4711 |

- **Oráculo:** subiu para 76,6%. Cada modelo diverso que entra soma ao ensemble.
- **McNemar pareado nessa divisão, média dos 10 contra o EP1:** só o EP1 acerta em 175 textos e só a média acerta em 221, com **p = 0,024**. A escolha foi feita na mesma divisão, então o valor é um pouco otimista.
- Os clássicos contribuem: sem eles, a média dos transformers cai para 0,4761.

**Decisão:** o candidato à entrega do EP2 é a **média simples dos 10 modelos**, retreinados em 100% do treino (modo `final`).

## 8. Simplificação do ensemble e estabilização de um único modelo, em 03-04/11

**Simplificação** (`python simplificar_ensemble.py` e `resultados/log_simplificacao.txt`): média simples de todos os 1.023 subconjuntos dos 10 componentes.
- Estimativa honesta (escolha numa metade, medida na outra): até 3 componentes, não há ganho sobre o EP1; com 5-10 componentes, o ganho fica entre +0,6 e +1,1 ponto.
- Qwen3, mmBERT, BERTimbau base e LogReg podem sair sem perda.

**Versão E:** 3× BERTimbau large (s42, s43 e s42 com rótulos suaves), 2× XLM-R large (normal e com rótulos suaves) e NB-SVM. Acurácia 0,4874, +1,6 ponto sobre o EP1, McNemar p = 0,004 na divisão 80/20. Foi a melhor com 6 componentes nesta divisão, então o valor é otimista.

**Estabilização de um único BERTimbau large, rodada 4** (`kaggle/saida_kaggle/run4/`): testamos se um modelo só substitui o ensemble de sementes.

| Variante | Semente 42 | Semente 43 | Com NB-SVM |
| :--- | :--- | :--- | :--- |
| Normal | 0,4653 | 0,4653 | 0,4660 / 0,4711 |
| EMA 0,999 | 0,4505 | 0,4668 | — |
| LLRD 0,9 | 0,4618 | 0,4575 | — |
| EMA + LLRD, 3 épocas | 0,4655 | 0,4630 | 0,4703 / 0,4660 |

**Conclusão:** um único transformer, mesmo estabilizado e com o NB-SVM, empata com o EP1 (~0,47). O ganho do EP2 vem de combinar várias execuções de transformers, ou seja, um deep ensemble, e foi essa a escolha: a versão E. A destilação desse ensemble num modelo só fica como trabalho futuro.

## 9. Ensemble final (versão E) e comparação no test1, em 04/11

`kaggle/envio_final` (notebook `ep2-final`) treinou os 5 transformers em 100% do treino, em 1h50.
- **Colapsos:** a proteção retreinou o BERTimbau large (semente 42 → 1042) e o XLM-R large (42 → 1042 → 2042).
- **Montagem:** `python nbsvm_final.py` e `python ensemble_final.py` geram `../test2-rotulado-ep2.xlsx` (**entrega**) e `../test1-rotulado-ep2.xlsx`.
- **Comparação:** `python ../comparar_teste1.py --ep2 test1-rotulado-ep2.xlsx` grava `../comparacao_teste1_ep1_ep2.txt`.

| test1 (gabarito do professor) | Acurácia | F1-macro |
| :--- | :--- | :--- |
| EP1 | 0,4967 | 0,4937 |
| EP2 (versão E) | 0,4889 | 0,4838 |

**McNemar:** só o EP1 acerta em 66 textos e só o EP2 acerta em 59; p = 0,59, sem diferença significativa.

**Análise** (feita depois, sem trocar o modelo):
- **Componentes isolados no test1:** de 0,4767 (NB-SVM) a 0,5000 (BERTimbau large s43). Todos estão dentro do mesmo intervalo.
- **Ruído do teste:** com 900 textos, o erro padrão da acurácia é de ±1,7 ponto (IC 95% de ±3,3). Para o McNemar dar p < 0,05 com ~125 pares discordantes, seriam necessários ~2,5 pontos de diferença. O ganho estimado no holdout (+0,6 a +1,6) não seria detectável nesse teste.
- **Teto de ruído dos rótulos:** 100 textos do test1 também aparecem no treino, e o rótulo majoritário do treino só coincide com o gabarito em **49%** deles. Mesmo memorizar perfeitamente os textos dá ~50%. O problema tem um teto baixo, e EP1 e EP2 já estão perto dele.
- Nos 100 textos repetidos, o EP1 acerta 47% e o EP2 45%; nos 800 inéditos, 50,0% e 49,4%.

## 10. LLMs maiores e decodificação balanceada, em 04/11

**Rodada 5 no Kaggle** (`kaggle/saida_kaggle/run5/`):

| Modelo | Acurácia no holdout | Tempo |
| :--- | :--- | :--- |
| Qwen3-4B (LoRA, 384 tokens, 2 épocas) | 0,4645 | 141 min por época; o modelo final também foi treinado (174 min por época) |
| Qwen3-8B (QLoRA 4 bits, 256 tokens, 1 época) | 0,4467 | 214 min |

Eles concordam 68-72% com os BERTs, mas **não somam ao ensemble**:

| Combinação | Acurácia |
| :--- | :--- |
| Versão E | 0,4874 |
| E + Qwen3-4B | 0,4816 |
| E + Qwen3-4B + Qwen3-8B | 0,4834 |
| Os 10 modelos anteriores + os dois Qwen | 0,4779 |

**Decodificação balanceada** (`decodificacao_balanceada.py`): força n/3 rótulos por classe, porque os testes parecem balanceados (900 = 3 × 300, 897 = 3 × 299). Avaliada em 200 amostras balanceadas de 900 textos do holdout, ela dá +0,3 ponto no EP1 e **−0,6 ponto na versão E**. Descartada.

**Poder estatístico de um teste com 900 textos** (2.000 sorteios balanceados do holdout):
- Mesmo com a vantagem que a versão E tem no holdout (+1,7 ponto, valor otimista porque a escolha foi feita nele), o McNemar só dá p < 0,05 em **38%** dos sorteios.
- O intervalo de 95% do ganho observado vai de −0,2 a +3,8 pontos.
- No test1, observamos −0,8 ponto. Isso indica que o ganho real é menor que o estimado no holdout, e que o test1 favoreceu o EP1: ele fez 0,4967 no test1 contra 0,4711 no holdout.

## 11. Métodos recentes da literatura, em 04/11

**Aprendizado confiante** (Northcutt et al., 2021; `cleanlab`), em `aprendizado_confiante.py` e `resultados/aprendizado_confiante.csv`:
- **Detecção:** previsões fora da dobra dentro dos 80% de treino, com CV interna de 5 folds agrupados e a média de LogReg w12c25 e NB-SVM. Sem vazamento do holdout.
- **Resultado:** o `cleanlab` marcou **44,6% do treino** (7.188 de 16.116) como rótulo suspeito, quase tudo o que o modelo erra.

| Treino | LogReg | NB-SVM | Média |
| :--- | :--- | :--- | :--- |
| Original | **0,4620** | **0,4600** | **0,4635** |
| Sem os suspeitos | 0,4499 | 0,4570 | 0,4560 |
| Suspeitos re-rotulados | 0,4517 | 0,4444 | 0,4479 |
| Peso 0,5 nos suspeitos | 0,4572 | 0,4595 | 0,4620 |

**Conclusão:** o ruído é subjetividade do rótulo, e não erro pontual de anotação. Limpar remove informação. Isso é coerente com o AlleNoise (AISTATS 2025) e o SaFER (ACL 2023): métodos para rótulos ruidosos falham quando o ruído é real.

**Encoders ModernBERT para português (2026), rodada 6** (`kaggle/saida_kaggle/run6/`), holdout 80/20:

| Modelo | Acurácia | Tempo por época |
| :--- | :--- | :--- |
| NorBERTo large, 1.024 tokens (1,9% de textos cortados) | 0,4630 | 19 min |
| NorBERTo large, 512 tokens | 0,4598 | 15 min |
| NorBERTo base | 0,4580 | 6,5 min |
| moBERTo | 0,4535 | 6,5 min |

- O contexto longo ajuda um pouco: +0,3 ponto no NorBERTo large.
- Nenhum supera o BERTimbau large (0,4653). Eles concordam ~71% com os outros modelos, mas **não somam**: E + NorBERTo large 1024 dá 0,4834, e E + todos os ModernBERT dá 0,4784, contra 0,4874 da versão E sozinha.

**Conclusão geral das seções 10-11:** modelos maiores (Qwen 4B/8B), mais novos (ModernBERT PT de 2026), contexto longo e limpeza de rótulos não superam a versão E. O teto de ~0,48 no holdout se mantém.

## 12. Análise de erros da versão E no test1, em 04/11 (aula 08)

`python analise_erros.py` gera `resultados/log_analise_erros.txt`, `resultados/analise_erros_caracteristicas.csv` e `resultados/analise_erros_amostra.csv` (60 erros, com a coluna `categoria_manual`).

**Matriz de confusão** (linhas = classe real; acurácia 0,4889):

| | prev. c1 | prev. c234 | prev. c5 | recall |
| :--- | :--- | :--- | :--- | :--- |
| c1 | 149 | 85 | 66 | 0,50 |
| c234 | 91 | 105 | 104 | 0,35 |
| c5 | 49 | 65 | 186 | 0,62 |

- 75% dos 460 erros envolvem a classe do meio (c234) e 25% são entre extremos (c1↔c5).
- Erros entre extremos indicam rótulos pouco ligados ao texto.

**Por característica:**
- **Confiança:**
  - textos com baixa confiança do ensemble (prob. máx. < 0,45; 36% do teste) têm acurácia 0,363;
  - textos com alta confiança (≥ 0,60; 17%) têm 0,755.
  - Ou seja, o ensemble sabe quando não sabe.
- **Textos repetidos do treino:** 0,450, e 0,435 quando os rótulos no treino divergem.
- Comprimento e padrões de resposta (negativa, encaminhamento, pedido de esclarecimento) mudam pouco a acurácia (de 0,48 a 0,54).

**Leitura manual de 60 erros** (10 por tipo de confusão), nas categorias do slide:

| Categoria | n | % | Categoria do slide |
| :--- | :--- | :--- | :--- |
| A. Conteúdo fora do texto (anexo, link, documento externo): o texto é só um "envelope" | 19 | 32% | definição do problema / falta de características |
| B. Negativa ou encaminhamento: a avaliação parece refletir o desfecho (atendido ou não), não a clareza | 20 | 33% | problema de rotulação |
| C. Resposta informativa com rótulo incoerente (o modelo erra com alta confiança, média de 0,58) | 6 | 10% | problema de rotulação |
| D. Modelo de resposta idêntico com rótulos divergentes (ex.: o mesmo texto padrão do Ministério da Fazenda) | 3 | 5% | problema de rotulação |
| E. Texto vazio ou truncado ("Prezado Senhor.", "esclarecemos que: 1.") | 2 | 3% | instância difícil |
| F. Fronteira ambígua entre classes vizinhas (resposta razoável, erro c234↔c1/c5) | 10 | 17% | instância difícil |

**Conclusão:** ~80% dos erros lidos (A-D) vêm da definição da tarefa e dos rótulos, e não de falha do modelo:
- o rótulo foi dado pelo cidadão olhando também o **anexo** e a **pergunta original**, e nenhum dos dois está no conjunto de dados;
- o rótulo parece misturar **clareza com satisfação** (respostas negativas claras recebem c1; encaminhamentos recebem c5).

Isso explica o teto de ~50% e por que nenhum modelo, nem os LLMs de 8B, passa dele. A característica que faltaria para melhorar é o **texto do pedido** (a pergunta do cidadão).

## 13. Validação cruzada de 5 folds: EP2 × EP1, em 05/11

São os mesmos 5 folds agrupados do EP1, com 20.092 textos. O fold 1 é o holdout já rodado; os folds 2 a 5 rodaram no Kaggle (kernels ep2-transformers v6 e ep2-pesado v2). O NB-SVM rodou localmente (`nbsvm_cv.py`). Para reproduzir: `cv_comparacao.py`, que grava `resultados/cv_comparacao.csv`. O McNemar usa o teste exato e compara cada modelo com o ensemble do EP1 nas 20.092 linhas.

| Modelo | Acurácia CV | Desvio | f1 | f2 | f3 | f4 | f5 | vs EP1 | Folds acima do EP1 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Ensemble do EP1 | 0,4705 | 0,004 | 0,4711 | 0,4695 | 0,4767 | 0,4634 | 0,4722 | — | — | — |
| BERTimbau large + rótulos suaves | 0,4678 | 0,009 | 0,4728 | 0,4611 | 0,4821 | 0,4584 | 0,4655 | −0,27 | 2/5 | 0,39 |
| BERTimbau large + EMA (s43) | 0,4648 | 0,005 | 0,4668 | 0,4567 | 0,4643 | 0,4642 | 0,4720 | −0,57 | 1/5 | 0,07 |
| BERTimbau large + EMA+LLRD, 3 épocas | 0,4635 | 0,007 | 0,4655 | 0,4702 | 0,4677 | 0,4512 | 0,4635 | −0,70 | 1/5 | 0,03 |
| **Versão E** | **0,4806** | 0,008 | 0,4874 | 0,4759 | 0,4920 | 0,4711 | 0,4774 | **+1,01** | **5/5** | **0,0001** |

Componentes da E na CV: bertimbau_large 0,4643; bertimbau_large_s43 0,4625; xlmr_large 0,4608; xlmr_large_suave 0,4456; NB-SVM 0,4517.

**Testes de robustez:**
- Só os folds 2 a 5, sem o fold usado para escolher a E: E 0,4789 contra EP1 0,4703 (+0,86 ponto, p = 0,0035).
- No fold 2, o xlmr_large_suave entrou em colapso parcial: nunca previu c1 e ficou com 0,37. A proteção não disparou, porque o limite é 85% numa só classe. Sem esse componente, a E de 5 componentes fica em 0,4790 (+0,85 ponto, p = 0,001). O colapso de um componente quase não afeta a média do ensemble.
- Houve 5 colapsos totais (100% numa só classe), todos retreinados automaticamente com a semente + 1000.

**Conclusões:**
1. Nenhum modelo isolado supera o ensemble do EP1 na CV. O melhor deles (rótulos suaves) empata (p = 0,39). Os ganhos dos isolados no holdout eram ruído do fold 1.
2. A versão E supera o EP1 nos 5 folds, com +1,0 ponto (+0,86 ponto sem o fold de seleção). Com 20 mil textos o resultado é significativo (p ≤ 0,004). No test1 (900 textos), essa mesma diferença não pode ser detectada, porque o poder estatístico fica em cerca de 30%.
3. O ganho vem da diversidade do ensemble (BERTimbau large + XLM-R + NB-SVM), e não de nenhum modelo isolado.
