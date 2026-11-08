# Versão E rodando localmente com os pesos salvos dos 5 transformers (Kaggle, notebook ep2-final v2, salvar_modelo=True)
# e o NB-SVM treinado aqui.
#
#   python versao_e_local.py --conferir   prevê test1/test2 localmente e compara:
#                                         (a) com as probabilidades que o próprio Kaggle gravou nesta rodada;
#                                         (b) com a versão E entregue (test2-rotulado-ep2.xlsx), que veio do retreino anterior.
#   python versao_e_local.py --lime       LIME na versão E completa (6 componentes) para os 6 textos do test1 da
#                                         análise de características (resultados/caracteristicas_locais.csv).
#
# Pesos esperados em kaggle/saida_kaggle/final_pesos/saida/<experimento>/modelo_semente<s>/

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = np.array(['c1', 'c234', 'c5'])
PASTA = 'kaggle/saida_kaggle/final_pesos/saida'
TRANSFORMERS = {'bertimbau_large_suave': 42, 'bertimbau_large': 42, 'bertimbau_large_s43': 43,
                'xlmr_large': 42, 'xlmr_large_suave': 42}
DISP = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def chave_texto(texto):
    return re.sub(r'\s+', ' ', texto).strip()


class Transformer:
    """Um componente da versão E: mesma tokenização do treino (ep2_treino.tokenizar: início + fim do texto)."""

    def __init__(self, nome, semente):
        self.nome = nome
        self.pasta = os.path.join(PASTA, f'final_{nome}', f'modelo_semente{semente}')
        with open(os.path.join(self.pasta, 'config_ep2.json'), encoding='utf-8') as f:
            self.cfg = json.load(f)
        assert self.cfg['classes'] == list(CLASSES)
        self.tok = AutoTokenizer.from_pretrained(self.pasta)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(self.pasta, dtype=torch.float16).eval()
        com = self.tok('palavra', add_special_tokens=True)['input_ids']
        sem = self.tok('palavra', add_special_tokens=False)['input_ids']
        i = next(i for i in range(len(com) - len(sem) + 1) if com[i:i + len(sem)] == sem)
        self.prefixo, self.sufixo = com[:i], com[i + len(sem):]

    def ids(self, textos):
        corpo = self.cfg['max_len'] - len(self.prefixo) - len(self.sufixo)
        fim = corpo - self.cfg['tokens_inicio']
        cod = self.tok([chave_texto(str(t)) for t in textos], add_special_tokens=False, truncation=False)['input_ids']
        return [self.prefixo + (t if len(t) <= corpo else t[:self.cfg['tokens_inicio']] + (t[-fim:] if fim > 0 else []))
                + self.sufixo for t in cod]

    @torch.no_grad()
    def proba(self, textos, lote=16):
        ids = self.ids(textos)
        self.modelo.to(DISP)
        saida = np.zeros((len(ids), 3), dtype=np.float32)
        ordem = np.argsort([len(i) for i in ids])
        for ini in range(0, len(ids), lote):
            sel = ordem[ini:ini + lote]
            m = max(len(ids[i]) for i in sel)
            x = torch.full((len(sel), m), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(sel), m), dtype=torch.long)
            for a, i in enumerate(sel):
                x[a, :len(ids[i])], att[a, :len(ids[i])] = torch.tensor(ids[i]), 1
            logits = self.modelo(input_ids=x.to(DISP), attention_mask=att.to(DISP)).logits
            saida[sel] = torch.softmax(logits.float(), -1).cpu().numpy()
        self.modelo.to('cpu')
        torch.cuda.empty_cache()
        return saida


def carregar():
    from analise_caracteristicas import NBSVM
    from src.data_loader import load_data
    data = load_data('../train.xlsx')
    nb = NBSVM(data['resp_text'].to_numpy(), data['clarity'].to_numpy())
    return [Transformer(n, s) for n, s in TRANSFORMERS.items()], nb


def conferir(trs, nb):
    for teste, arq in [('test1', '../ep1/data/test1.xlsx'), ('test2', '../test2.xlsx')]:
        textos = pd.read_excel(arq)['resp_text'].fillna('').astype(str).to_numpy()
        soma = nb.predict_proba(textos).astype(float)
        print(f'\n{teste} ({len(textos)} textos)')
        for t in trs:
            p = t.proba(textos)
            kag = pd.read_csv(os.path.join(PASTA, f'final_{t.nome}',
                                           f'final_{teste}_semente{TRANSFORMERS[t.nome]}.csv'))[[f'p_{c}' for c in CLASSES]].to_numpy()
            print(f'  {t.nome:<24} local x Kaggle: máx. dif. prob. {np.abs(p - kag).max():.4f}, '
                  f'mesmo rótulo em {np.mean(p.argmax(1) == kag.argmax(1)):.1%}')
            soma += p
        novo = CLASSES[soma.argmax(1)]
        pd.DataFrame(soma / 6, columns=[f'p_{c}' for c in CLASSES]).to_csv(
            f'finais/ensemble/probabilidades_{teste}_pesos_salvos.csv', index_label='indice')
        entregue = pd.read_excel(f'../{teste}-rotulado-ep2.xlsx')['clarity'].to_numpy()
        print(f'  versão E (pesos salvos) x versão E entregue: mesmo rótulo em {np.mean(novo == entregue):.1%} '
              f'({int((novo != entregue).sum())} rótulos diferentes)')
        if teste == 'test1':
            y = pd.read_excel('../test1-rotulado-para-teste-estatistico.xlsx')['clarity'].to_numpy()
            print(f'  acurácia no test1: pesos salvos {np.mean(novo == y):.4f} | entregue {np.mean(entregue == y):.4f}')


def lime(trs, nb):
    from lime.lime_text import LimeTextExplainer
    locais = pd.read_csv('resultados/caracteristicas_locais.csv')
    gab = pd.read_excel('../test1-rotulado-para-teste-estatistico.xlsx')
    textos1 = gab['resp_text'].fillna('').astype(str).to_numpy()

    def proba_e(textos):
        return (nb.predict_proba(textos) + sum(t.proba(textos) for t in trs)) / 6

    explicador = LimeTextExplainer(class_names=list(CLASSES), random_state=42)
    linhas = []
    for _, r in locais.iterrows():
        i = int(r['indice_test1'])
        k = int(np.where(CLASSES == r['previsto_E'])[0][0])
        ex = explicador.explain_instance(textos1[i], proba_e, labels=[k], num_features=10, num_samples=500)
        lista = ex.as_list(label=k)
        a_favor = {t.lower() for t, w in lista if w > 0}
        nb_lime = {t.split(' (')[0].lower() for t in str(r['lime_nbsvm']).split('; ') if '(+' in t}
        bert_lime = {t.split(' (')[0].lower() for t in str(r.get('lime_bert', '')).split('; ') if '(+' in t}
        linhas.append({'indice_test1': i, 'motivo': r['motivo'], 'real': r['real'], 'previsto_E': r['previsto_E'],
                       'lime_versao_E': '; '.join(f'{t} ({w:+.3f})' for t, w in lista),
                       'comum_com_lime_nbsvm': ', '.join(sorted(a_favor & nb_lime)),
                       'comum_com_lime_bertimbau_ep1': ', '.join(sorted(a_favor & bert_lime))})
        print(f"\ntest1[{i}] | {r['motivo']} | real {r['real']} | E {r['previsto_E']}")
        for c in ['lime_versao_E', 'comum_com_lime_nbsvm', 'comum_com_lime_bertimbau_ep1']:
            print(f'  {c}: {linhas[-1][c]}')
    pd.DataFrame(linhas).to_csv('resultados/lime_versao_e.csv', index=False, encoding='utf-8-sig')
    print('\n-> resultados/lime_versao_e.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conferir', action='store_true')
    ap.add_argument('--lime', action='store_true')
    args = ap.parse_args()
    trs, nb = carregar()
    if args.conferir:
        conferir(trs, nb)
    if args.lime:
        lime(trs, nb)


if __name__ == '__main__':
    main()
