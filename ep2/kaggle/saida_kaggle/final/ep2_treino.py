# Fine-tuning de transformers para {c1, c234, c5}: encoders (BERTimbau, XLM-R, mmBERT...) e LLMs com LoRA/QLoRA.
#
# Configurado por um JSON (só as chaves que mudam em relação a PADRAO). O notebook ep2_transformers.ipynb grava
# este arquivo no Kaggle e roda um experimento por GPU em paralelo; localmente:
#   python ep2_treino.py config.json
#
# Modos:
#   holdout -> treina nos 80% e prevê os 20% de divisao_80_20.csv (mesma divisão dos clássicos do EP2)
#   cv      -> 5 folds de folds_5.csv (os mesmos das OOF do EP1); gera as probabilidades out-of-fold
#   final   -> treina em 100% do train.xlsx e prevê test1.xlsx (sem rótulos) e test2.xlsx
# Saída em <saida>/<experimento>/: <divisao>_semente<s>_epoca<e>.csv, final_<teste>_semente<s>.csv,
# historico.csv, config.json.

import glob
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PADRAO = {
    'experimento': 'teste',
    'modelo': 'neuralmind/bert-large-portuguese-cased',
    'tipo': 'encoder',            # 'encoder', 'llm_lora' (base em fp16) ou 'llm_qlora' (base em 4 bits)
    'modos': ['holdout'],
    'folds_cv': None,             # cv: None = os 5 folds; ou uma lista, ex. [1, 2]
    'max_len': 512,               # tokens; textos longos mantêm o início e o fim
    'tokens_inicio': 128,         # tokens mantidos do início; o restante vem do fim
    'epocas': 2,
    'lr': 1.5e-5,
    'lote': 8,                    # tamanho do lote na GPU
    'acumulacao': 2,              # lote efetivo = lote x acumulacao
    'warmup': 0.1,
    'sementes': [42],
    'rotulos_suaves': False,      # alvo = distribuição dos rótulos do texto no treino (textos repetidos divergentes)
    'gradient_checkpointing': False,
    'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
    'ema': None,                  # ex. 0.999: avalia/usa a média móvel exponencial dos pesos (um modelo só, mais estável)
    'llrd': 1.0,                  # ex. 0.9: lr decrescente nas camadas de baixo (camada i recebe lr * llrd^(L-i))
    'max_tentativas': 3,          # se o modelo colapsar (>85% das predições numa classe), retreina com outra semente
    'amostra': None,              # ex. 400: usa só N textos (teste rápido do código)
    'pasta_dados': None,          # None = procura train.xlsx em /kaggle/input, /content e na pasta atual
    'saida': None,                # None = /kaggle/working/saida (Kaggle) ou ./saida
}
CLASSES = ['c1', 'c234', 'c5']
ROTULO_PARA_ID = {c: i for i, c in enumerate(CLASSES)}
cfg = {}
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def achar_arquivo(pasta, nome):
    """Procura o arquivo na pasta de dados e, se não estiver lá, em qualquer subpasta de /kaggle/input ou /content
    (no Kaggle, arquivos enviados em versões diferentes do Dataset podem ficar em pastas diferentes)."""
    caminho = os.path.join(pasta, nome)
    if os.path.exists(caminho):
        return caminho
    achados = glob.glob(f'/kaggle/input/**/{nome}', recursive=True) + glob.glob(f'/content/**/{nome}', recursive=True)
    assert achados, f'{nome} não encontrado em {pasta} nem em /kaggle/input'
    return achados[0]


def log(*args):
    print(*args, flush=True)


def chave_texto(texto):
    """Normalização do EP1: textos que só diferem em espaços/quebras de linha são o mesmo texto."""
    return re.sub(r'\s+', ' ', texto).strip()


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def tokens_especiais(tokenizer):
    """Tokens especiais que o modelo põe antes e depois do texto (ex.: [CLS]/[SEP], <s>/</s>, nenhum)."""
    com = tokenizer('palavra', add_special_tokens=True)['input_ids']
    sem = tokenizer('palavra', add_special_tokens=False)['input_ids']
    for i in range(len(com) - len(sem) + 1):
        if com[i:i + len(sem)] == sem:
            return com[:i], com[i + len(sem):]
    raise ValueError('não foi possível identificar os tokens especiais do tokenizer')


def tokenizar(textos, tokenizer):
    """Tokeniza; textos longos mantêm tokens_inicio tokens do início e o restante do fim."""
    prefixo, sufixo = tokens_especiais(tokenizer)
    corpo = cfg['max_len'] - len(prefixo) - len(sufixo)
    tokens_fim = corpo - cfg['tokens_inicio']
    codificados = tokenizer([chave_texto(str(t)) for t in textos], add_special_tokens=False, truncation=False)['input_ids']
    ids = []
    for tokens in codificados:
        if len(tokens) > corpo:
            tokens = tokens[:cfg['tokens_inicio']] + (tokens[-tokens_fim:] if tokens_fim > 0 else [])
        ids.append(prefixo + tokens + sufixo)
    return ids


def alvos_treino(textos, rotulos):
    """Matriz (n x 3) de alvos. Com rotulos_suaves, cada linha é a distribuição dos rótulos do seu texto
    (normalizado) dentro do próprio conjunto de treino; senão, one-hot."""
    one_hot = np.eye(len(CLASSES), dtype=np.float32)[rotulos]
    if not cfg['rotulos_suaves']:
        return one_hot
    chaves = np.array([chave_texto(t) for t in textos])
    dist = pd.DataFrame(one_hot).groupby(chaves).transform('mean').to_numpy(dtype=np.float32)
    log(f'  rótulos suaves: {(dist.max(1) < 1).sum()} linhas com rótulos divergentes')
    return dist


def criar_lotes(ids, tamanho_lote, embaralhar, rng=None):
    """Agrupa exemplos de tamanho parecido no mesmo lote (menos padding = treino mais rápido)."""
    indices = np.arange(len(ids))
    if embaralhar:
        rng.shuffle(indices)
        tamanho_bloco = tamanho_lote * 50
        blocos = [indices[i:i + tamanho_bloco] for i in range(0, len(indices), tamanho_bloco)]
        indices = np.concatenate([sorted(b, key=lambda i: len(ids[i])) for b in blocos])
    else:
        indices = np.array(sorted(indices, key=lambda i: len(ids[i])))
    lotes = [indices[i:i + tamanho_lote] for i in range(0, len(indices), tamanho_lote)]
    if embaralhar:
        rng.shuffle(lotes)
    return lotes


def montar_lote(ids, lote, id_padding):
    maior = max(len(ids[i]) for i in lote)
    input_ids = torch.full((len(lote), maior), id_padding, dtype=torch.long)
    mascara = torch.zeros((len(lote), maior), dtype=torch.long)
    for j, i in enumerate(lote):
        n = len(ids[i])
        input_ids[j, :n] = torch.tensor(ids[i], dtype=torch.long)
        mascara[j, :n] = 1
    return input_ids.to(dispositivo), mascara.to(dispositivo)


def arg_dtype(dtype):
    # 'dtype' substituiu 'torch_dtype' no transformers 4.56
    from packaging.version import Version
    import transformers
    return {'dtype' if Version(transformers.__version__) >= Version('4.56') else 'torch_dtype': dtype}


def criar_modelo(tokenizer):
    kwargs = dict(num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)), label2id=ROTULO_PARA_ID)
    if cfg['tipo'] == 'encoder':
        # pesos mestres em float32; o forward roda em fp16 com autocast
        modelo = AutoModelForSequenceClassification.from_pretrained(cfg['modelo'], **kwargs).float()
        if cfg['gradient_checkpointing']:
            modelo.gradient_checkpointing_enable()
        return modelo.to(dispositivo)

    from peft import LoraConfig, TaskType, get_peft_model
    if cfg['tipo'] == 'llm_qlora':
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        modelo = AutoModelForSequenceClassification.from_pretrained(
            cfg['modelo'], quantization_config=quant, **arg_dtype(torch.float16), device_map={'': 0}, **kwargs)
        modelo = prepare_model_for_kbit_training(
            modelo, use_gradient_checkpointing=cfg['gradient_checkpointing'],
            gradient_checkpointing_kwargs={'use_reentrant': False})
    elif cfg['tipo'] == 'llm_lora':
        modelo = AutoModelForSequenceClassification.from_pretrained(cfg['modelo'], **arg_dtype(torch.float16), **kwargs)
        if cfg['gradient_checkpointing']:
            modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            modelo.enable_input_require_grads()
    else:
        raise ValueError(cfg['tipo'])
    modelo.config.pad_token_id = tokenizer.pad_token_id
    modelo = get_peft_model(modelo, LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                                               lora_dropout=cfg['lora_dropout'], target_modules='all-linear'))
    # parâmetros treináveis (LoRA + cabeça de classificação) em float32, exigência do GradScaler
    for p in modelo.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    modelo.print_trainable_parameters()
    return modelo.to(dispositivo) if cfg['tipo'] == 'llm_lora' else modelo


def criar_otimizador(modelo):
    """AdamW sem weight decay em bias/LayerNorm. Com llrd < 1, cada camada do encoder recebe lr * llrd^(distância
    até o topo): embeddings recebem o menor lr; a cabeça de classificação, o lr cheio."""
    sem_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
    treinaveis = [(n, p) for n, p in modelo.named_parameters() if p.requires_grad]
    camadas = [int(m.group(1)) for n, _ in treinaveis for m in [re.search(r'\.layers?\.(\d+)\.', n)] if m]
    n_camadas = max(camadas) + 1 if camadas else 0

    def profundidade(nome):
        m = re.search(r'\.layers?\.(\d+)\.', nome)
        if m:
            return int(m.group(1)) + 1
        return 0 if 'embed' in nome else n_camadas + 1   # embeddings embaixo; pooler/cabeça no topo

    grupos = {}
    for n, p in treinaveis:
        lr = cfg['lr'] * cfg['llrd'] ** (n_camadas + 1 - profundidade(n)) if cfg['llrd'] < 1 else cfg['lr']
        decay = 0.0 if any(s in n for s in sem_decay) else 0.01
        grupos.setdefault((lr, decay), []).append(p)
    if cfg['llrd'] < 1:
        lrs = sorted({lr for lr, _ in grupos})
        log(f'  LLRD {cfg["llrd"]}: {n_camadas} camadas, lr de {lrs[0]:.2e} (embeddings) a {lrs[-1]:.2e} (cabeça)')
    return torch.optim.AdamW([{'params': ps, 'lr': lr, 'weight_decay': d} for (lr, d), ps in grupos.items()])


def criar_agendador(otimizador, passos_totais):
    passos_warmup = int(passos_totais * cfg['warmup'])

    def fator(passo):
        if passo < passos_warmup:
            return passo / max(1, passos_warmup)
        return max(0.0, (passos_totais - passo) / max(1, passos_totais - passos_warmup))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, fator)


@torch.no_grad()
def prever_probabilidades(modelo, ids, id_padding):
    """Matriz de probabilidades (n x 3) na ordem original dos textos."""
    modelo.eval()
    probs = np.zeros((len(ids), len(CLASSES)), dtype=np.float32)
    for lote in criar_lotes(ids, cfg['lote'] * 2, embaralhar=False):
        input_ids, mascara = montar_lote(ids, lote, id_padding)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = modelo(input_ids=input_ids, attention_mask=mascara).logits
        probs[lote] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def salvar_probs(probs, caminho, indices, rotulos=None, fold=None):
    df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
    df.insert(0, 'indice', indices)
    if fold is not None:
        df.insert(1, 'fold', fold)
    if rotulos is not None:
        df['rotulo'] = rotulos
    df['previsto'] = np.array(CLASSES)[probs.argmax(1)]
    df.to_csv(caminho, index=False)


@torch.no_grad()
def trocar_pesos(modelo, novos):
    """Copia os tensores de `novos` para os parâmetros do modelo e devolve os valores antigos."""
    antigos = {}
    for n, p in modelo.named_parameters():
        if n in novos:
            antigos[n] = p.detach().clone()
            p.copy_(novos[n].to(p.dtype))
    return antigos


def treinar(ids_treino, alvos, tokenizer, semente, ids_val=None, y_val=None, ao_fim_da_epoca=None):
    """Treina pelo nº fixo de épocas; se houver validação, avalia ao fim de cada época."""
    fixar_semente(semente)
    rng = np.random.default_rng(semente)
    id_padding = tokenizer.pad_token_id
    modelo = criar_modelo(tokenizer)
    otimizador = criar_otimizador(modelo)
    lotes_por_epoca = math.ceil(len(ids_treino) / cfg['lote'])
    agendador = criar_agendador(otimizador, math.ceil(lotes_por_epoca / cfg['acumulacao']) * cfg['epocas'])
    escalador = torch.amp.GradScaler('cuda')
    alvos = torch.tensor(alvos, dtype=torch.float32)
    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    # EMA: cópia em float32 dos pesos treináveis, atualizada a cada passo do otimizador
    ema = ({n: p.detach().clone().float() for n, p in modelo.named_parameters() if p.requires_grad}
           if cfg['ema'] else None)
    passo = 0

    historico = []
    for epoca in range(1, cfg['epocas'] + 1):
        modelo.train()
        inicio, perda_acumulada, nao_finitas = time.time(), 0.0, 0
        lotes = criar_lotes(ids_treino, cfg['lote'], embaralhar=True, rng=rng)
        otimizador.zero_grad()
        for n_lote, lote in enumerate(lotes, start=1):
            input_ids, mascara = montar_lote(ids_treino, lote, id_padding)
            alvo = alvos[torch.as_tensor(lote)].to(dispositivo)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = modelo(input_ids=input_ids, attention_mask=mascara).logits
            perda = F.cross_entropy(logits.float(), alvo)
            if torch.isfinite(perda):
                escalador.scale(perda / cfg['acumulacao']).backward()
                perda_acumulada += perda.item()
            else:
                nao_finitas += 1  # overflow em fp16: o lote é descartado
            if n_lote % cfg['acumulacao'] == 0 or n_lote == len(lotes):
                escalador.unscale_(otimizador)
                torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
                escalador.step(otimizador)
                escalador.update()
                otimizador.zero_grad()
                agendador.step()
                passo += 1
                if ema is not None:
                    d = min(cfg['ema'], (1 + passo) / (10 + passo))   # decaimento menor no início
                    with torch.no_grad():
                        for n, p in modelo.named_parameters():
                            if n in ema:
                                ema[n].mul_(d).add_(p.detach().float(), alpha=1 - d)
            if n_lote % 200 == 0:
                decorrido = time.time() - inicio
                log(f'  época {epoca} | lote {n_lote}/{len(lotes)} | perda {perda_acumulada / n_lote:.4f} '
                    f'| ~{decorrido / n_lote * (len(lotes) - n_lote) / 60:.1f} min restantes')

        registro = {'semente': semente, 'epoca': epoca, 'perda_treino': perda_acumulada / len(lotes),
                    'lotes_nao_finitos': nao_finitas, 'minutos': (time.time() - inicio) / 60}
        if ids_val is not None:
            if ema is not None:
                originais = trocar_pesos(modelo, ema)
            probs_val = prever_probabilidades(modelo, ids_val, id_padding)
            if ema is not None:
                trocar_pesos(modelo, originais)
            previsto = probs_val.argmax(1)
            registro['acuracia'] = float((previsto == y_val).mean())
            registro['distribuicao_prevista'] = str(np.bincount(previsto, minlength=3).tolist())
            if ao_fim_da_epoca is not None:
                ao_fim_da_epoca(epoca, probs_val)
        historico.append(registro)
        log('  ' + ' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in registro.items()))
        if nao_finitas > 0.05 * len(lotes):
            log(f'  ATENÇÃO: {nao_finitas} lotes com perda não finita (overflow em fp16); reduza o lr ou troque o modelo')
    if ema is not None:
        trocar_pesos(modelo, ema)   # o modelo devolvido (e usado no modo final) é o da média móvel
    return modelo, historico


def colapsou(modelo, ids_amostra, id_padding):
    """O BERT large às vezes diverge e passa a prever uma única classe; detecta isso numa amostra do treino."""
    previsto = prever_probabilidades(modelo, ids_amostra, id_padding).argmax(1)
    fracao = np.bincount(previsto, minlength=len(CLASSES)).max() / len(previsto)
    return fracao > 0.85, fracao


def treinar_com_verificacao(ids_treino, alvos, tokenizer, semente, ids_val=None, y_val=None, ao_fim_da_epoca=None):
    """Treina; se o modelo colapsar, descarta e retreina com outra semente (até max_tentativas vezes)."""
    rng = np.random.default_rng(0)
    amostra = [ids_treino[i] for i in rng.choice(len(ids_treino), min(1000, len(ids_treino)), replace=False)]
    for tentativa in range(cfg['max_tentativas']):
        semente_usada = semente + 1000 * tentativa
        modelo, historico = treinar(ids_treino, alvos, tokenizer, semente_usada, ids_val, y_val, ao_fim_da_epoca)
        ruim, fracao = colapsou(modelo, amostra, tokenizer.pad_token_id)
        for h in historico:
            h['semente_usada'] = semente_usada
        if not ruim:
            return modelo, historico
        if tentativa == cfg['max_tentativas'] - 1:
            log(f'  ATENÇÃO: o modelo colapsou em todas as {cfg["max_tentativas"]} tentativas; mantendo o último')
            return modelo, historico
        log(f'  COLAPSO: {fracao:.0%} das predições numa só classe (semente {semente_usada}); retreinando')
        del modelo
        torch.cuda.empty_cache()


def main(caminho_config):
    with open(caminho_config, encoding='utf-8') as f:
        cfg.update(PADRAO)
        cfg.update(json.load(f))
    assert dispositivo.type == 'cuda', 'sem GPU'

    pasta = cfg['pasta_dados']
    if pasta is None:
        achados = (glob.glob('/kaggle/input/**/train.xlsx', recursive=True)
                   + glob.glob('/content/**/train.xlsx', recursive=True) + glob.glob('train.xlsx'))
        assert achados, 'train.xlsx não encontrado'
        pasta = os.path.dirname(achados[0])
    raiz = cfg['saida'] or ('/kaggle/working/saida' if os.path.exists('/kaggle/working') else os.path.abspath('saida'))
    saida = os.path.join(raiz, cfg['experimento'])
    os.makedirs(saida, exist_ok=True)
    with open(os.path.join(saida, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    import transformers
    log(f"Experimento {cfg['experimento']}: {cfg['modelo']} ({cfg['tipo']}) | modos {cfg['modos']} | saída {saida}")
    log(f'GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | transformers {transformers.__version__}')

    treino = pd.read_excel(achar_arquivo(pasta, 'train.xlsx'))
    treino['resp_text'] = treino['resp_text'].fillna('').astype(str)
    divisao = pd.read_csv(achar_arquivo(pasta, 'divisao_80_20.csv'))
    folds = pd.read_csv(achar_arquivo(pasta, 'folds_5.csv'))
    assert (divisao['indice'].to_numpy() == np.arange(len(treino))).all()
    assert (folds['indice'].to_numpy() == np.arange(len(treino))).all()
    if cfg['amostra']:
        n = cfg['amostra']
        treino, divisao, folds = treino.iloc[:n].copy(), divisao.iloc[:n].copy(), folds.iloc[:n].copy()

    tokenizer = AutoTokenizer.from_pretrained(cfg['modelo'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    textos = treino['resp_text'].tolist()
    rotulos_texto = treino['clarity'].to_numpy()
    y_todos = treino['clarity'].map(ROTULO_PARA_ID).to_numpy()
    ids_todos = tokenizar(textos, tokenizer)
    tamanhos = np.array([len(i) for i in ids_todos])
    log(f"Tokens por texto: mediana {np.median(tamanhos):.0f}, p90 {np.percentile(tamanhos, 90):.0f}, "
        f"{(tamanhos >= cfg['max_len']).mean():.1%} truncados")
    historicos = []

    def guardar_historico(historico, nome):
        historicos.extend({**h, 'divisao': nome} for h in historico)
        pd.DataFrame(historicos).to_csv(os.path.join(saida, 'historico.csv'), index=False)

    def rodar_divisao(nome, idx_tr, idx_av, semente, fold=None):
        ids_tr, ids_av = [ids_todos[i] for i in idx_tr], [ids_todos[i] for i in idx_av]
        alvos = alvos_treino([textos[i] for i in idx_tr], y_todos[idx_tr])

        def ao_fim_da_epoca(epoca, probs):
            salvar_probs(probs, os.path.join(saida, f'{nome}_semente{semente}_epoca{epoca}.csv'), idx_av,
                         rotulos=rotulos_texto[idx_av], fold=fold)

        modelo, historico = treinar_com_verificacao(ids_tr, alvos, tokenizer, semente, ids_av, y_todos[idx_av], ao_fim_da_epoca)
        guardar_historico(historico, nome)
        del modelo
        torch.cuda.empty_cache()

    for semente in cfg['sementes']:
        if 'holdout' in cfg['modos']:
            log(f'\n========== HOLDOUT 80/20 | semente {semente} ==========')
            rodar_divisao('holdout', np.where(divisao['conjunto'] == 'treino')[0],
                          np.where(divisao['conjunto'] == 'avaliacao')[0], semente)
        if 'cv' in cfg['modos']:
            for f in (cfg['folds_cv'] or sorted(folds['fold'].unique())):
                log(f'\n========== CV fold {f} | semente {semente} ==========')
                rodar_divisao(f'cv_fold{f}', np.where(folds['fold'] != f)[0], np.where(folds['fold'] == f)[0],
                              semente, fold=f)
        if 'final' in cfg['modos']:
            log(f'\n========== FINAL (100% do treino) | semente {semente} ==========')
            modelo, historico = treinar_com_verificacao(ids_todos, alvos_treino(textos, y_todos), tokenizer, semente)
            guardar_historico(historico, 'final')
            for nome in ['test1', 'test2']:
                teste = pd.read_excel(achar_arquivo(pasta, f'{nome}.xlsx'))
                probs = prever_probabilidades(modelo, tokenizar(teste['resp_text'].fillna('').astype(str), tokenizer),
                                              tokenizer.pad_token_id)
                salvar_probs(probs, os.path.join(saida, f'final_{nome}_semente{semente}.csv'), np.arange(len(teste)))
                log(f'{nome}: {len(teste)} textos | distribuição prevista '
                    + str(pd.Series(np.array(CLASSES)[probs.argmax(1)]).value_counts().to_dict()))
            del modelo
            torch.cuda.empty_cache()
    log(f'\nConcluído. Arquivos: {sorted(os.listdir(saida))}')


if __name__ == '__main__':
    main(sys.argv[1])
