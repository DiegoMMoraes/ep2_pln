# Análise de características: versão E

Esta análise segue a aula 08 (*Análise de características e de erros*), com interpretação **global** e **local** do modelo, e complementa a [análise de erros](ANALISE_ERROS.md).

## 1. Como foi feita

```bash
cd ep2
python analise_caracteristicas.py     # --sem-bert pula o LIME no BERTimbau
```

**Saídas**

| Arquivo | Conteúdo |
| :--- | :--- |
| `resultados/log_analise_caracteristicas.txt` | log completo |
| `resultados/caracteristicas_globais.csv` | 300 n-gramas de maior peso por classe, com frequência, distribuição por classe e tipo |
| `resultados/caracteristicas_locais.csv` | explicação de 6 textos do test1 (NB-SVM e LIME) |

**Modelo de apoio (slide 20).** Os 5 transformers da versão E não são interpretáveis diretamente (slide 19). Seguindo a sugestão da aula, usamos como **modelo de apoio** o **NB-SVM**, que é linear e já é um componente da própria versão E. Ele foi treinado com 100% do treino, com a mesma configuração da entrega:
- n-gramas binários de palavras 1–2, sem acentos;
- LogReg um-contra-todos com C = 0,25.

O peso de um n-grama numa classe é o coeficiente da LogReg multiplicado pela razão de log-contagem do Naive Bayes. É o equivalente ao `eli5.show_weights` (slides 22–24 e 26). Não usamos o eli5 porque ele não é compatível com o scikit-learn 1.9, mas o cálculo dos pesos é o mesmo.

**Etapas**
1. **Interpretação global:** ranking dos n-gramas de maior peso em c1, c234 e c5.
2. **Atalhos e características espúrias** (slides 27 e 45, exemplo dos cadernos da FSP). Cada um dos 300 n-gramas de maior peso por classe foi classificado em:
   - **nome próprio/sigla:** contém uma palavra que aparece com inicial maiúscula em pelo menos 80% das ocorrências no meio da frase;
   - **número/data;**
   - **saudação/assinatura;**
   - **contato/URL/canal;**
   - **template:** menos da metade das ocorrências vem de textos distintos;
   - **conteúdo:** o restante.
3. **Fidelidade do modelo de apoio.** Medimos a concordância e o kappa entre o NB-SVM e a versão E no test1 e na validação cruzada de 5 folds, para saber o quanto a explicação do NB-SVM vale para o ensemble.
4. **Interpretação local** (slides 25–26), em 6 textos do test1:
   - os 3 acertos de maior confiança, um por classe;
   - o erro de maior confiança de cada uma das categorias A, B e C da análise de erros.

   Para cada texto, mostramos três explicações:
   - a contribuição de cada n-grama no NB-SVM (equivalente ao `eli5.show_predictions`);
   - o **LIME** (Ribeiro et al., 2016; 500 perturbações) aplicado ao NB-SVM;
   - o **LIME** aplicado ao **BERTimbau do EP1**. Os pesos dos transformers da versão E ficaram no Kaggle, que só devolveu as probabilidades. Por isso usamos esse BERTimbau como substituto do componente BERT. Ele acerta 0,4833 no test1.

## 2. Interpretação global

**Os 10 n-gramas de maior peso por classe** (NB-SVM; "docs" é o número de textos do treino que contêm o n-grama)

| # | c1 | docs | c234 | docs | c5 | docs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | sra comunicamos | 8 | transmito documento | 8 | **odila** | 46 |
| 2 | popular informa | 19 | no 2016 | 8 | arquivo quadro | 7 |
| 3 | **progep ufal** | 8 | projetos da | 8 | **senhora odila** | 45 |
| 4 | seja area | 5 | via email | 9 | 2018 anexo | 7 |
| 5 | **luis cesar** | 57 | questionamento acerca | 6 | **senhora thais** | 6 |
| 6 | valores distribuidos | 35 | oficio em | 6 | pedido ouvida | 7 |
| 7 | **ana gabriela** | 4 | **adriel** | 4 | desejada sem | 5 |
| 8 | **supes** | 5 | **www2 ifmg** | 17 | de finalizacao | 7 |
| 9 | compreender sua | 11 | **cgfse digef** | 4 | **cristiane mendes** | 14 |
| 10 | responsavel existem | 7 | **milene** | 5 | encaminhamos no | 10 |

Em negrito estão nomes de pessoas e siglas de órgãos. A lista completa, com os 300 n-gramas por classe, está no CSV.

**Tipos de característica entre as 300 de maior peso por classe**

| Classe | conteúdo | nome próprio/sigla | número/data | contato/URL | saudação/assinatura | template |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| c1 | 56% | 27% | 12% | 2% | 1% | 2% |
| c234 | 60% | 26% | 9% | 4% | 1% | 0% |
| c5 | 52% | **38%** | 7% | 1% | 2% | 0% |

A detecção de nomes e siglas é conservadora e deixa passar siglas como `coap` e `igovti`, então esses percentuais são um **limite inferior**.

**O que a interpretação global mostra**
- **O modelo de apoio quase não aprende "clareza".** Entre 40% e 48% das características mais fortes de cada classe são nomes, siglas, números, contatos ou saudações. Os n-gramas de conteúdo que aparecem também são muito específicos: a mediana é de **4 textos** por n-grama entre os 300 de maior peso. Não aparecem termos genéricos de clareza, como "segue a informação" ou "não foi possível".
- **Efeito do solicitante.** O caso mais forte é a cidadã "Odila": **46 textos no treino, todos rotulados c5**. "Luis Cesar" tem 57 textos, 88% deles em c1. Como quem dá a nota é o próprio cidadão, **o rótulo depende de quem pergunta**. O modelo aprende esse padrão pelo nome que aparece na saudação da resposta ("Prezada Senhora Odila").
  - Esse é exatamente o tipo de atalho do exemplo da FSP (slide 27): uma característica que acerta, mas não tem relação com a tarefa.
  - Ele reforça a conclusão da análise de erros de que o rótulo mede a **satisfação do cidadão**, e não só a clareza do texto.
- **O atalho funciona dentro deste conjunto.** Os mesmos solicitantes reaparecem nos testes: 7,9% dos textos do test1 e 7,2% dos do test2 contêm algum dos nomes ou siglas do top 300. Nesses textos, a versão E acerta **0,662**; nos demais, **0,474**.
  - Para a nota do EP2 isso ajuda, porque o test2 vem da mesma fonte.
  - Para um uso real, com cidadãos novos, isso é sobreajuste ao solicitante (slide 8). A acurácia esperada seria a dos textos sem nomes, ~0,47.
- **Templates pesam pouco.** Só 0–2% das características vêm de textos repetidos. A deduplicação e a divisão agrupada por texto evitaram que o modelo dependesse de modelos de resposta idênticos.

## 3. Fidelidade do modelo de apoio

| Par de modelos | Conjunto | Concordância | Kappa |
| :--- | :--- | :--- | :--- |
| NB-SVM × versão E | test1 | 0,734 | 0,601 |
| NB-SVM × versão E | CV 5 folds (20.092) | 0,717 | 0,575 |
| BERTimbau large suave × versão E | CV 5 folds | 0,806 | 0,708 |
| NB-SVM × BERTimbau large suave | CV 5 folds | 0,602 | **0,402** |
| *Versão E × gabarito* | *test1 / CV* | *0,489 / 0,481* | *0,233 / 0,221* |

- O NB-SVM concorda com a versão E em cerca de 72% dos casos (kappa ≈ 0,58), uma concordância moderada. A explicação do modelo de apoio vale para a maior parte das decisões do ensemble, mas não para todas. Como os slides dizem, os dois modelos "não são a mesma coisa".
- O NB-SVM e o BERTimbau concordam pouco entre si (kappa 0,40), ou seja, **erram em lugares diferentes**. É essa diversidade que explica por que a média dos dois tipos de modelo (+1,0 ponto na CV) supera cada um isolado.
- O kappa de qualquer modelo com o gabarito é baixo (~0,22). Isso é coerente com o teto de concordância de 49% nos textos repetidos, discutido na análise de erros.

## 4. Interpretação local

Mostramos as 4 palavras de maior peso a favor da classe prevista pela versão E. As listas completas, com 10 palavras e os pesos contra, estão em `resultados/caracteristicas_locais.csv`.

| test1 | Caso | Real → E (confiança) | NB-SVM (pesos) | LIME no NB-SVM | LIME no BERTimbau |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 142 | acerto c1 | c1 → c1 (0,85) | luis cesar, sr luis, luis encaminhamos, gerente executiva | Luis, Cesar, Executiva, Gerente | informação, 12527, bancária, interesse |
| 836 | acerto c234 | c234 → c234 (0,61) | 2016 abono, 2011 setembro, anexa planilha, vagas oriundas | setembro, anexa, vagas, 2016 | vagas, anexa, Portaria, convocações |
| 453 | acerto c5 | c5 → c5 (0,88) | parecer solicitado, solicitacao encaminhamos, sic procuradoria, dr marcelo | Escalante, solicitado, Dr, Marcelo | cópia, Dr, anexo, encaminhamos |
| 402 | erro A (anexo/link) | c234 → c5 (0,57) | uploads sites, encaminhamos anexo, wp, regimento do | uploads, anexo, sites, Encaminhamos | anexo, Encaminhamos, Estrutura, inovacao |
| 654 | erro B (negativa) | c1 → c5 (0,59) | por ja, vaga, de biblioteca, conforme art | vaga, de, Prezada, ajustar | biblioteca, nomeação, não, Prezada |
| 610 | erro C (rótulo incoerente) | c1 → c5 (0,66) | epp, epp cnpj, 229, ltda epp | EPP, 7724, LTDA, 229 | Concessionário, Infraero, anterior, 2017 |

**O que a interpretação local mostra**
- **Os dois modelos olham para coisas diferentes.** O NB-SVM decide por nomes e números: o nome do solicitante no texto 142, o nome do procurador no 453, CNPJ e "EPP" no 610. O BERTimbau decide mais por palavras de conteúdo e de ação, como "informação", "cópia", "anexo", "encaminhamos" e "Concessionário". Esse padrão local confirma o kappa baixo entre os dois (0,40).
- **No texto 142 (acerto c1), o NB-SVM acerta pelo motivo errado.** Ele decide pelo nome "Luis Cesar", e não pelo conteúdo da resposta. É o efeito do solicitante visto num caso concreto.
- **Erro A (texto 402).** Os dois modelos associam "encaminhamos anexo" a c5. Como a resposta só remete ao anexo, a avaliação depende do documento, que nenhum dos modelos vê.
- **Erro C (texto 610).** É uma lista de concessionários com CNPJ, ou seja, informativa. O cidadão deu c1, e os dois modelos previram c5 guiados pelo formato: números de CNPJ para o NB-SVM, "Concessionário" e "Infraero" para o BERTimbau. Nada no texto indica a clareza percebida pelo cidadão.

## 5. Conclusões

1. **Características espúrias.** Pelo menos ~30% das características mais fortes do modelo de apoio são nomes de pessoas e siglas de órgãos, e outros ~10% são números. O caso mais claro é o **efeito do solicitante**: o mesmo cidadão tende a dar sempre a mesma nota. Esse é o "atalho" que a aula 08 pede para investigar.
2. **O atalho ajuda no teste, mas não generaliza.** Ele melhora a acurácia nos textos do test1 com nomes conhecidos (0,66 contra 0,47). Com solicitantes novos, a acurácia esperada cai para o patamar dos textos sem nomes.
3. **Diversidade do ensemble.** O modelo de apoio (NB-SVM) e o BERTimbau usam evidências diferentes (nomes e números contra palavras de conteúdo) e concordam pouco (kappa 0,40). Isso explica o ganho da média dos dois na versão E.
4. **Coerência com a análise de erros.** Nenhum dos modelos encontra no texto um sinal estável de clareza. O que se aprende é **quem pergunta**, **que órgão responde** e o **formato** da resposta (anexo, lista, encaminhamento). Isso apoia a conclusão de que o rótulo mistura clareza e satisfação, e de que o teto de ~50% vem dos dados.

## 6. Limitações

- **O modelo de apoio não é a versão E.** A concordância entre os dois é moderada (kappa 0,58), então a explicação vale aproximadamente.
- **O LIME não foi aplicado à versão E.** Ele rodou num substituto (o BERTimbau base do EP1), porque os pesos dos transformers da E não foram salvos.
- **As heurísticas de tipo de característica são aproximadas.** A de nomes e siglas, em particular, subestima a proporção real.
- **O LIME é instável com 500 amostras e explica só 6 textos.** Os pesos locais são ilustrativos, não estatísticos.
- **Teste do atalho em produção:** uma forma de confirmar o efeito seria remover nomes próprios dos textos (anonimização) e medir a queda de acurácia. Não fizemos esse teste.
