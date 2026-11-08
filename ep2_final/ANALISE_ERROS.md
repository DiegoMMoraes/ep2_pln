# Análise de erros: versão final no test1

Esta análise segue a aula 08 (*Análise de características e de erros*). O objetivo é inspecionar as instâncias classificadas incorretamente e identificar **classes de problemas**, começando pelas mais frequentes. O slide 53 propõe três categorias:
- instâncias difíceis de julgar ou classificar;
- problemas de rotulação ou de definição do problema;
- necessidade de características adicionais.

## 1. Dados e modelo analisados

- **Modelo:** a versão final. É a média simples das probabilidades de 3 BERTimbau large, 2 XLM-R large e 1 NB-SVM, todos treinados com 100% do `train.xlsx`.
- **Conjunto:** test1 (900 textos, 300 por classe). Os rótulos verdadeiros vêm do gabarito do professor (`dados/test1-gabarito.xlsx`).
  - Usamos o test1 porque é o único conjunto rotulado que o modelo final não viu no treino.
  - O test1 não foi usado para escolher modelo nem hiperparâmetros; a análise é feita depois da escolha.
- **Entradas do script:**
  - previsões: `entrega/test1-rotulado-ep2.xlsx`;
  - probabilidades do ensemble: `saidas/ensemble/probabilidades_test1.csv`;
  - treino: `dados/train.xlsx`, usado para detectar textos repetidos.

## 2. Como foi feita

A análise tem quatro etapas. As três primeiras são automáticas, feitas pelo script `analise_erros.py`. A quarta é a leitura manual de uma amostra de erros.

```bash
cd ep2_final
python analise_erros.py
```

**Saídas do script**

| Arquivo | Conteúdo |
| :--- | :--- |
| `resultados/log_analise_erros.txt` | matriz de confusão, recall por classe, tipos de erro e acurácia por característica |
| `resultados/analise_erros_caracteristicas.csv` | tabela de acurácia por característica do texto |
| `resultados/analise_erros_amostra.csv` | 60 erros sorteados, com o texto (até 600 caracteres) e a coluna `categoria_manual` |

### Etapa 1: matriz de confusão e tipos de erro

O script compara a previsão com o gabarito e calcula:
- a matriz de confusão;
- o recall de cada classe;
- a frequência de cada tipo de erro (real→previsto).

Os erros são separados em dois grupos:
- os que **envolvem a classe do meio** (c234): confusão entre vizinhos, esperada numa escala ordinal;
- os **entre extremos** (c1↔c5): indicam que o rótulo tem pouca relação com o texto.

### Etapa 2: acurácia por característica do texto

Cada texto é marcado com características detectadas automaticamente. Depois o script compara a acurácia do grupo que tem a característica com a do resto. As características foram escolhidas para representar as categorias do slide:

| Característica | Como é detectada | Hipótese testada |
| :--- | :--- | :--- |
| Repetido do treino | o texto normalizado (`chave_texto`: espaços colapsados) aparece no treino | problema de rotulação |
| Repetido com rótulos divergentes | o mesmo texto tem mais de um rótulo no treino | problema de rotulação |
| Texto curto ou longo | menos de 30 ou mais de 300 palavras | instância difícil |
| Baixa ou alta confiança | probabilidade máxima do ensemble < 0,45 ou ≥ 0,60 | instância difícil ou possível erro de rótulo |
| Remete a anexo ou link | regex: `anexo`, `segue a resposta`, `http`, `link`... | falta de características (a informação está fora do texto) |
| Pede esclarecimento | regex: `reformul`, `esclareç`, `especifique`... | definição do problema |
| Nega ou diz não ter a informação | regex: `não possui`, `inexist`, `sigilo`, `indeferido`... | rótulo mistura clareza e satisfação |
| Encaminha a outro órgão | regex: `encaminh`, `órgão competente`... | rótulo mistura clareza e satisfação |

### Etapa 3: amostra estratificada de erros

Para a leitura manual não ficar concentrada no erro mais comum, o script sorteia (semente 42) **10 erros de cada um dos 6 tipos de confusão**, num total de 60. Cada linha do CSV traz:
- a classe real, a prevista e a confiança;
- o número de palavras;
- se o texto é repetido do treino;
- as marcações das regex;
- o texto.

### Etapa 4: leitura e categorização manual

Lemos os 60 textos e atribuímos a cada um uma categoria, registrada na coluna `categoria_manual` do CSV. As categorias A a F não foram definidas antes da leitura. Elas surgiram dos próprios textos (codificação aberta) e depois foram associadas às três categorias do slide.

## 3. Resultados

### Matriz de confusão

Acurácia de 0,4889 (440 de 900). Nas linhas está a classe real.

| | prev. c1 | prev. c234 | prev. c5 | recall |
| :--- | :--- | :--- | :--- | :--- |
| **c1** | 149 | 85 | 66 | 0,50 |
| **c234** | 91 | 105 | 104 | 0,35 |
| **c5** | 49 | 65 | 186 | 0,62 |

**Tipos de erro (460 erros)**

| real→previsto | % dos erros |
| :--- | :--- |
| c234→c5 | 22,6% |
| c234→c1 | 19,8% |
| c1→c234 | 18,5% |
| c1→c5 | 14,3% |
| c5→c234 | 14,1% |
| c5→c1 | 10,7% |

- **75%** dos erros envolvem a classe do meio (c234), que é a mais difícil (recall de 0,35).
- **25%** são entre extremos (c1↔c5). Neles, o rótulo e o texto parecem dizer coisas opostas.

### Acurácia por característica

| Característica | n | Acurácia no grupo | Acurácia no resto |
| :--- | :--- | :--- | :--- |
| Baixa confiança (< 0,45) | 328 | **0,363** | 0,561 |
| Alta confiança (≥ 0,60) | 151 | **0,755** | 0,435 |
| Repetido do treino | 100 | 0,450 | 0,494 |
| Repetido com rótulos divergentes | 69 | 0,435 | 0,493 |
| Texto curto (< 30 palavras) | 140 | 0,493 | 0,488 |
| Texto longo (> 300 palavras) | 82 | 0,537 | 0,484 |
| Nega ou diz não ter a informação | 179 | 0,542 | 0,476 |
| Encaminha a outro órgão | 217 | 0,507 | 0,483 |
| Pede esclarecimento | 237 | 0,498 | 0,486 |
| Remete a anexo ou link* | 552 | 0,476 | 0,509 |

\* A regex de anexo/link ficou ampla demais: marcou 552 dos 900 textos. Por isso essa linha não serve para conclusões, e o efeito do anexo foi avaliado na leitura manual (categoria A).

**O que a tabela mostra**
- **O ensemble sabe quando não sabe.** A acurácia vai de 0,36 nos textos de baixa confiança a 0,76 nos de alta confiança.
- **Textos repetidos do treino são mais difíceis.** Ter visto o texto no treino não ajuda, porque o mesmo texto recebe rótulos diferentes.
  - Nos 100 textos do test1 que também estão no treino, o rótulo majoritário do treino coincide com o gabarito em só **49%** dos casos.
  - Isso funciona como um teto de concordância entre rotuladores, e ele fica perto da acurácia dos modelos.
- Comprimento e tipo de resposta mudam pouco a acurácia, que fica entre 0,48 e 0,54.

### Leitura manual dos 60 erros

| Categoria | n | % | Confiança média | Categoria do slide |
| :--- | :--- | :--- | :--- | :--- |
| **A.** Conteúdo fora do texto: a resposta só remete a anexo, link ou documento externo, e o texto é um "envelope" | 19 | 32% | 0,44 | falta de características / definição do problema |
| **B.** Negativa ou encaminhamento: o rótulo parece refletir o desfecho (atendido ou não), e não a clareza | 20 | 33% | 0,46 | problema de rotulação |
| **C.** Resposta informativa com rótulo incoerente: o modelo erra com alta confiança | 6 | 10% | 0,58 | problema de rotulação |
| **D.** Modelo de resposta idêntico com rótulos divergentes (ex.: o mesmo texto padrão do Ministério da Fazenda) | 3 | 5% | 0,51 | problema de rotulação |
| **E.** Texto vazio ou truncado ("Prezado Senhor.", "esclarecemos que: 1.") | 2 | 3% | 0,37 | instância difícil |
| **F.** Fronteira ambígua entre classes vizinhas: resposta razoável, erro entre c234 e c1 ou c5 | 10 | 17% | 0,45 | instância difícil |

**Categorias por tipo de erro**

| real→previsto | A | B | C | D | E | F |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| c1→c234 | 2 | 2 | 2 | 0 | 1 | 3 |
| c1→c5 | 3 | 5 | 2 | 0 | 0 | 0 |
| c234→c1 | 7 | 3 | 0 | 0 | 0 | 0 |
| c234→c5 | 4 | 2 | 0 | 0 | 0 | 4 |
| c5→c1 | 0 | 3 | 2 | 3 | 1 | 1 |
| c5→c234 | 3 | 5 | 0 | 0 | 0 | 2 |

**Padrões**
- Nos erros entre extremos (c1↔c5), dominam as categorias B, C e D, ligadas ao rótulo: 15 dos 20.
- A categoria F, de ambiguidade real, só aparece em erros que envolvem c234.
- A categoria A se concentra em c234→c1. Quando a resposta só diz "segue em anexo", o cidadão avalia o anexo, que o modelo não vê.

## 4. Conclusões

1. **Cerca de 80% dos erros lidos (categorias A a D) vêm da definição da tarefa e dos rótulos, e não de falha do modelo.**
   - O cidadão avaliou a resposta vendo também o **anexo** e a **própria pergunta**, e nenhum dos dois está no conjunto de dados.
   - O rótulo parece misturar **clareza com satisfação**: respostas negativas claras recebem c1, e encaminhamentos recebem c5.
2. **Só cerca de 20% são instâncias difíceis de verdade** (categorias E e F), concentradas na fronteira com c234.
3. Isso explica o **teto de cerca de 50%**. Nenhum modelo passou dele: nem os BERT large, nem os LLMs de 4B e 8B (Qwen3), nem o aprendizado confiante (cleanlab).
4. **A característica que faltaria é o texto do pedido do cidadão.** Com a pergunta, daria para julgar se a resposta é clara para *aquela* pergunta. Ela não está disponível no conjunto de dados.
5. **Uso prático da confiança:** como a acurácia passa de 0,75 quando a confiança é alta, o modelo pode servir como triagem. Ele classificaria sozinho os casos de alta confiança e mandaria os demais para revisão humana.

## 5. Limitações

- **Amostra pequena:** são 60 erros, 10 por tipo. Os percentuais por categoria são indicativos, com margem de erro de cerca de ±12 pontos.
- **Amostra não proporcional:** ela é balanceada por tipo de erro, e não pela frequência real de cada tipo. Os erros entre extremos ficam super-representados (20 de 60, contra 25% do total).
- **Um único avaliador:** a categorização foi feita uma vez, por uma pessoa, sem medir a concordância entre avaliadores.
- **Regex:** os padrões automáticos são heurísticos; o de anexo/link, em particular, foi amplo demais.
