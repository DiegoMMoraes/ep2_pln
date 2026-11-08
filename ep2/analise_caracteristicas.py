# Análise de características (aula 08): interpretação global e local da versão E.
#
# Os transformers da versão E não são interpretáveis diretamente. Seguindo o slide 20, usamos como MODELO DE APOIO o
# NB-SVM, que é linear e já é um componente da própria versão E.
#   1. Global: n-gramas de maior peso por classe no NB-SVM treinado com 100% do treino
#      (peso = coeficiente da LogReg x razão de log-contagem do NB, como em eli5.show_weights).
#   2. Atalhos/características espúrias (slides 27 e 45): entre as 300 características de maior peso por classe, a
#      fração de nomes próprios/siglas, números, saudações/assinaturas, contatos/URLs e de "template" (n-gramas que aparecem quase só em
#      textos repetidos).
#   3. Fidelidade do modelo de apoio: concordância (kappa) entre o NB-SVM e a versão E no test1 e na CV de 5 folds.
#   4. Local (slides 25-26): para 6 textos do test1 (3 acertos de alta confiança, 1 por classe, e 3 erros das
#      categorias A, B e C da análise de erros):
#      - contribuição de cada n-grama no NB-SVM (como eli5.show_predictions);
#      - LIME (Ribeiro et al., 2016) no NB-SVM e no BERTimbau do EP1. O BERTimbau é o único transformer cujos pesos
#        estão disponíveis localmente; os da versão E ficaram no Kaggle. Ele funciona como substituto do componente
#        BERT da versão E.
#
# Uso: python analise_caracteristicas.py [--sem-bert]
# Saídas: resultados/caracteristicas_globais.csv, resultados/caracteristicas_locais.csv,
#         resultados/log_analise_caracteristicas.txt

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

from busca_holdout import CLASSES, softmax
from nbsvm_final import C, vetorizador
from src.data_loader import load_data
from src.deduplicacao import chave_texto

PASTA_BERT = '../ep1/modelos/bertimbau/final_2ep/final'
N_AMOSTRAS_LIME = 500

SAUDACAO = r'^(prezad|senhor|senhora|sr|sra|caro|cara|atenciosamente|cordialmente|grato|grata|agradec|obrigad|' \
           r'ola|bom dia|boa tarde|respeitosamente|saudac)'
CONTATO = r'(http|www|\bbr\b|\bcom\b|gov br|email|e mail|telefone|\btel\b|ramal|endereco|cep|\bsic\b|esic|' \
          r'e sic|fala br|ouvidoria)'


class NBSVM:
    """NB-SVM um-contra-todos (o mesmo de busca_holdout.treinar_nbsvm), guardando os pesos para interpretação."""

    def __init__(self, X, y):
        self.vet = vetorizador().fit(X)
        B = self.vet.transform(X)
        self.r, self.clfs = [], []
        for c in CLASSES:
            pos = np.asarray(B[y == c].sum(0)).ravel() + 1
            neg = np.asarray(B[y != c].sum(0)).ravel() + 1
            r = np.log((pos / pos.sum()) / (neg / neg.sum()))
            self.r.append(r)
            self.clfs.append(LogisticRegression(C=C, max_iter=3000).fit(B.multiply(r).tocsr(), (y == c).astype(int)))
        self.nomes = self.vet.get_feature_names_out()
        self.pesos = np.vstack([clf.coef_.ravel() * r for clf, r in zip(self.clfs, self.r)])   # 3 x n_features

    def predict_proba(self, textos):
        B = self.vet.transform(list(textos))
        return softmax(np.column_stack([clf.decision_function(B.multiply(r).tocsr())
                                        for clf, r in zip(self.clfs, self.r)]))

    def contribuicoes(self, texto, k):
        """Contribuição de cada n-grama presente no texto para a classe k (binário: contribuição = peso)."""
        idx = self.vet.transform([texto]).nonzero()[1]
        return sorted(((self.nomes[j], self.pesos[k, j]) for j in idx), key=lambda t: -t[1])


def vocabulario_maiusculo(textos):
    """Palavras (sem acento, minúsculas) que aparecem com inicial maiúscula ou toda em maiúsculas em >= 80% das
    ocorrências no meio da frase: nomes próprios de pessoas, órgãos e siglas."""
    from sklearn.feature_extraction.text import strip_accents_unicode
    maiusc, total = {}, {}
    for t in textos:
        for m in re.finditer(r'(?<=[a-zà-ú0-9,;] )([A-Za-zÀ-ú]{2,})', str(t)):
            w = m.group(1)
            k = strip_accents_unicode(w.lower())
            total[k] = total.get(k, 0) + 1
            maiusc[k] = maiusc.get(k, 0) + w[0].isupper()
    return {k for k, n in total.items() if n >= 3 and maiusc[k] / n >= 0.8}


PROPRIOS = set()


def tipo_espurio(termo):
    if any(w in PROPRIOS for w in termo.split()):
        return 'nome próprio/sigla'
    if re.search(r'\d', termo):
        return 'número/data'
    if re.search(SAUDACAO, termo):
        return 'saudação/assinatura'
    if re.search(CONTATO, termo):
        return 'contato/URL/canal'
    return ''


def bert_proba_fn():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, '../ep1')
    from finetune_bertimbau import tokenizar
    with open(os.path.join(PASTA_BERT, 'config_treino.json'), encoding='utf-8') as f:
        config = json.load(f)
    disp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(PASTA_BERT)
    modelo = AutoModelForSequenceClassification.from_pretrained(PASTA_BERT).to(disp).eval()
    assert [modelo.config.id2label[i] for i in range(3)] == list(CLASSES)

    def proba(textos, lote=32):
        ids = tokenizar(list(textos), tok, config['max_len'], config['tokens_inicio'])
        saida = np.zeros((len(ids), 3), dtype=np.float32)
        ordem = np.argsort([len(i) for i in ids])
        for ini in range(0, len(ids), lote):
            sel = ordem[ini:ini + lote]
            m = max(len(ids[i]) for i in sel)
            x = torch.full((len(sel), m), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(sel), m), dtype=torch.long)
            for a, i in enumerate(sel):
                x[a, :len(ids[i])] = torch.tensor(ids[i])
                att[a, :len(ids[i])] = 1
            with torch.no_grad(), torch.autocast(device_type=disp.type, dtype=torch.float16,
                                                 enabled=disp.type == 'cuda'):
                logits = modelo(input_ids=x.to(disp), attention_mask=att.to(disp)).logits
            saida[sel] = torch.softmax(logits.float(), -1).cpu().numpy()
        return saida
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sem-bert', action='store_true')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 120)

    data = load_data('../train.xlsx')
    X, y = data['resp_text'].to_numpy(), data['clarity'].to_numpy()
    print(f'Treinando o NB-SVM (modelo de apoio) com {len(y)} textos...', flush=True)
    nb = NBSVM(X, y)
    PROPRIOS.update(vocabulario_maiusculo(X))

    # ---------- 1 e 2. Interpretação global e atalhos ----------
    B = nb.vet.transform(X)
    df_total = np.asarray((B > 0).sum(0)).ravel()
    chaves = pd.Series(X).apply(chave_texto)
    unicos = ~chaves.duplicated().to_numpy()
    df_unicos = np.asarray((B[unicos] > 0).sum(0)).ravel()
    por_classe = {c: np.asarray((B[y == c] > 0).sum(0)).ravel() for c in CLASSES}

    linhas, resumo = [], []
    for k, c in enumerate(CLASSES):
        top = np.argsort(-nb.pesos[k])[:300]
        for pos, j in enumerate(top, 1):
            linhas.append({
                'classe': c, 'posicao': pos, 'ngrama': nb.nomes[j], 'peso': round(nb.pesos[k, j], 4),
                'docs': int(df_total[j]), 'frac_textos_unicos': round(df_unicos[j] / df_total[j], 3),
                **{f'pct_{d}': round(por_classe[d][j] / df_total[j], 3) for d in CLASSES},
                'tipo_espurio': tipo_espurio(nb.nomes[j]) or ('template' if df_unicos[j] / df_total[j] < 0.5 else ''),
            })
    glob_ = pd.DataFrame(linhas)
    glob_.to_csv('resultados/caracteristicas_globais.csv', index=False, encoding='utf-8-sig')

    print('\n=== 1. Interpretação global: 25 n-gramas de maior peso por classe (NB-SVM) ===')
    for c in CLASSES:
        t = glob_[(glob_.classe == c) & (glob_.posicao <= 25)]
        print(f'\n--- {c} ---')
        print(t[['posicao', 'ngrama', 'peso', 'docs', 'pct_c1', 'pct_c234', 'pct_c5', 'frac_textos_unicos',
                 'tipo_espurio']].to_string(index=False))

    print('\n=== 2. Características possivelmente espúrias entre as 300 de maior peso por classe ===')
    for c in CLASSES:
        t = glob_[glob_.classe == c]
        cont = t['tipo_espurio'].replace('', 'conteúdo').value_counts(normalize=True).round(3)
        resumo.append({'classe': c, **cont.to_dict()})
    print(pd.DataFrame(resumo).fillna(0).to_string(index=False))
    print('(template = n-grama em que menos da metade das ocorrências vem de textos distintos)')

    # ---------- 3. Fidelidade do modelo de apoio ----------
    gab = pd.read_excel('../test1-rotulado-para-teste-estatistico.xlsx')
    textos1 = gab['resp_text'].fillna('').astype(str).to_numpy()
    y1 = gab['clarity'].to_numpy()
    p_e = pd.read_csv('finais/ensemble/probabilidades_test1.csv')[[f'p_{c}' for c in CLASSES]].to_numpy()
    p_nb = nb.predict_proba(textos1)
    prev_e, prev_nb = CLASSES[p_e.argmax(1)], CLASSES[p_nb.argmax(1)]

    import cv_comparacao as cv
    _, ycv, fold = cv.oof_csv('bertimbau')
    P = {k: cv.oof_transformer(k, s, len(ycv), fold) for k, s in cv.TRANSFORMERS.items()}
    P['nbsvm'] = cv.oof_csv('nbsvm')[0]
    e_cv = CLASSES[sum(P[k] for k in cv.VERSAO_E).argmax(1)]
    nb_cv = CLASSES[P['nbsvm'].argmax(1)]
    bert_cv = CLASSES[P['bertimbau_large_suave'].argmax(1)]

    print('\n=== 3. Fidelidade do modelo de apoio (NB-SVM) à versão E ===')
    print(f'test1:  concordância NB-SVM x E {np.mean(prev_nb == prev_e):.3f} (kappa {cohen_kappa_score(prev_nb, prev_e):.3f})'
          f' | acurácia NB-SVM {np.mean(prev_nb == y1):.4f}, E {np.mean(prev_e == y1):.4f}')
    print(f'CV 5 folds: concordância NB-SVM x E {np.mean(nb_cv == e_cv):.3f} (kappa {cohen_kappa_score(nb_cv, e_cv):.3f});'
          f' BERTimbau large suave x E {np.mean(bert_cv == e_cv):.3f} (kappa {cohen_kappa_score(bert_cv, e_cv):.3f});'
          f' NB-SVM x BERTimbau {np.mean(nb_cv == bert_cv):.3f} (kappa {cohen_kappa_score(nb_cv, bert_cv):.3f})')
    print(f'Referência: kappa da versão E com o gabarito no test1 {cohen_kappa_score(prev_e, y1):.3f};'
          f' na CV {cohen_kappa_score(e_cv, ycv):.3f}')

    # ---------- 4. Interpretação local ----------
    conf = p_e.max(1)
    escolhidos = []
    for c in CLASSES:
        cand = np.where((y1 == c) & (prev_e == c))[0]
        escolhidos.append((int(cand[np.argmax(conf[cand])]), f'acerto de alta confiança ({c})'))
    amostra = pd.read_csv('resultados/analise_erros_amostra.csv')
    for cat in ['A', 'B', 'C']:
        a = amostra[amostra['categoria_manual'].astype(str).str.startswith(cat)].sort_values('confianca').iloc[-1]
        escolhidos.append((int(a['indice_test1']), f'erro categoria {a["categoria_manual"]}'))

    explicador = LimeTextExplainer(class_names=list(CLASSES), random_state=42)
    bert = None if args.sem_bert else bert_proba_fn()
    if bert is not None:
        p_bert = bert(textos1)
        print(f'\n(BERTimbau do EP1 no test1: acurácia {np.mean(CLASSES[p_bert.argmax(1)] == y1):.4f}; '
              'confere com a entrega do EP1 se for ~0,49)')

    print(f'\n=== 4. Interpretação local de {len(escolhidos)} textos do test1 (LIME com {N_AMOSTRAS_LIME} amostras) ===')
    locais = []
    for i, motivo in escolhidos:
        texto = textos1[i]
        k = int(p_e[i].argmax())
        reg = {'indice_test1': i, 'motivo': motivo, 'real': y1[i], 'previsto_E': CLASSES[k],
               'confianca_E': round(conf[i], 3), 'previsto_NBSVM': prev_nb[i], 'texto': texto[:700]}
        contrib = nb.contribuicoes(texto, k)
        reg['nbsvm_pesos_a_favor'] = '; '.join(f'{t} ({w:+.2f})' for t, w in contrib[:8])
        reg['nbsvm_pesos_contra'] = '; '.join(f'{t} ({w:+.2f})' for t, w in contrib[::-1][:5])
        modelos = [('lime_nbsvm', nb.predict_proba)] + ([('lime_bert', bert)] if bert is not None else [])
        tops = {}
        for nome, fn in modelos:
            ex = explicador.explain_instance(texto, fn, labels=[k], num_features=10, num_samples=N_AMOSTRAS_LIME)
            lista = ex.as_list(label=k)
            tops[nome] = {t.lower() for t, w in lista if w > 0}
            reg[nome] = '; '.join(f'{t} ({w:+.3f})' for t, w in lista)
        if len(tops) == 2:
            reg['palavras_em_comum_a_favor'] = ', '.join(sorted(tops['lime_nbsvm'] & tops['lime_bert']))
        locais.append(reg)
        print(f'\n--- test1[{i}] | {motivo} | real {y1[i]} | E previu {CLASSES[k]} ({conf[i]:.2f}) | '
              f'NB-SVM previu {prev_nb[i]}')
        print('Texto:', re.sub(r'\s+', ' ', texto)[:400])
        for campo in ['nbsvm_pesos_a_favor', 'nbsvm_pesos_contra', 'lime_nbsvm', 'lime_bert', 'palavras_em_comum_a_favor']:
            if campo in reg:
                print(f'  {campo}: {reg[campo]}')
    pd.DataFrame(locais).to_csv('resultados/caracteristicas_locais.csv', index=False, encoding='utf-8-sig')
    print('\n-> resultados/caracteristicas_globais.csv, resultados/caracteristicas_locais.csv')


if __name__ == '__main__':
    main()
