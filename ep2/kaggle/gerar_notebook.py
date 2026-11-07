# Gera kaggle/ep2_transformers.ipynb: configuração (lista de experimentos) + ep2_treino.py (via %%writefile) +
# lançador que roda um experimento por GPU em paralelo (T4 x2 no Kaggle = 2 experimentos ao mesmo tempo).
#
# Uso: python kaggle/gerar_notebook.py

import json
import os

PASTA = os.path.dirname(os.path.abspath(__file__))
celulas = []


def md(texto):
    celulas.append({'cell_type': 'markdown', 'metadata': {}, 'source': texto.strip('\n').splitlines(keepends=True)})


def code(texto):
    celulas.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                    'source': texto.strip('\n').splitlines(keepends=True)})


md('''
# EP2: fine-tuning de transformers (encoders e LLMs com LoRA / QLoRA)

Edite só a lista `EXPERIMENTOS` na próxima célula. Cada experimento é um dicionário com as chaves que mudam em relação
ao padrão (ver `PADRAO` em `ep2_treino.py`). O lançador roda **um experimento por GPU, em paralelo**: com o
acelerador **GPU T4 x2**, dois experimentos ao mesmo tempo; os demais esperam na fila.

| Preset | `modelo` | `tipo` | `lr` | `lote` x `acumulacao` | Tempo estimado do holdout (2 épocas, T4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BERTimbau large | `neuralmind/bert-large-portuguese-cased` | encoder | 1.5e-5 | 8 x 2 | ~25 min (já rodado: 0,4653) |
| XLM-RoBERTa large | `FacebookAI/xlm-roberta-large` | encoder | 1e-5 | 8 x 2 | ~30 min |
| mmBERT base (2025) | `jhu-clsp/mmBERT-base` | encoder | 3e-5 | 8 x 2 | ~10 min |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | llm_lora | 1e-4 | 4 x 4 | ~2 h |
| Qwen3 4B | `Qwen/Qwen3-4B` | llm_qlora | 1e-4 | 4 x 4 | ~4 h (`max_len` 256: ~2 h) |
| Qwen3 8B | `Qwen/Qwen3-8B` | llm_qlora | 1e-4 | 2 x 8 | ~8 h (`max_len` 256, 1 época: ~2-3 h) |

Tempos são estimativas; o log de cada experimento mostra os minutos restantes. LLMs: use
`gradient_checkpointing: True`. `llm_qlora` carrega o modelo em 4 bits (bitsandbytes, instalado na célula seguinte).

**Modos** (`modos`): `holdout` (80/20, comparação rápida), `cv` (5 folds, probabilidades out-of-fold para o
stacking; `folds_cv: [1, 2]` roda só alguns folds) e `final` (100% do treino, prevê test1 e test2).
**Opções**: `sementes: [42, 43, 44]`; `rotulos_suaves: True` (alvo = distribuição dos rótulos de textos repetidos);
`amostra: 400` para um teste rápido do código.

**Kaggle**: Dataset com os 5 arquivos de `dados/` > Add Input; Accelerator **GPU T4 x2**; Internet **On**;
**Save Version > Save & Run All (Commit)**. Saída em `/kaggle/working/saida/<experimento>/` (inclui `log.txt`).
''')

code('''
EXPERIMENTOS = [
    dict(experimento='xlmr_large', modelo='FacebookAI/xlm-roberta-large', lr=1e-5),
    dict(experimento='qwen3_1.7b', modelo='Qwen/Qwen3-1.7B', tipo='llm_lora', lr=1e-4, lote=4, acumulacao=4,
         gradient_checkpointing=True),
    dict(experimento='mmbert_base', modelo='jhu-clsp/mmBERT-base', lr=3e-5),
    dict(experimento='bertimbau_large_suave', modelo='neuralmind/bert-large-portuguese-cased', lr=1.5e-5,
         rotulos_suaves=True),
]
COMUM = dict(modos=['holdout'])   # aplicado a todos os experimentos (cada um pode sobrescrever)
''')

code('''
# Dependências: peft (LoRA) e bitsandbytes (QLoRA), só se algum experimento precisar
import importlib.util, subprocess, sys
tipos = {e.get('tipo', 'encoder') for e in EXPERIMENTOS}
faltando = [p for p, precisa in [('peft', bool(tipos & {'llm_lora', 'llm_qlora'})), ('bitsandbytes', 'llm_qlora' in tipos)]
            if precisa and importlib.util.find_spec(p) is None]
if faltando:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *faltando], check=True)
# O peft recusa versões antigas do torchao (o Kaggle traz a 0.10); sem ele instalado, o peft funciona normalmente
if tipos & {'llm_lora', 'llm_qlora'} and importlib.util.find_spec('torchao') is not None:
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version('torchao')) < Version('0.16'):
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=True)
        print('torchao antigo removido (incompatível com o peft)')
print('dependências ok', faltando or '')
''')

with open(os.path.join(PASTA, 'ep2_treino.py'), encoding='utf-8') as f:
    code('%%writefile ep2_treino.py\n' + f.read())

code('''
# Lançador: um experimento por GPU, em paralelo; mostra o progresso a cada minuto
import json, os, subprocess, sys, time
import pandas as pd
import torch

RAIZ = '/kaggle/working/saida' if os.path.exists('/kaggle/working') else os.path.abspath('saida')
os.makedirs(RAIZ, exist_ok=True)
# confere os dados antes de gastar GPU
import glob
achados = glob.glob('/kaggle/input/**/train.xlsx', recursive=True) + glob.glob('/content/**/train.xlsx', recursive=True) + glob.glob('train.xlsx')
assert achados, 'train.xlsx não encontrado: adicione o Dataset (Add Input)'
PASTA_DADOS = os.path.dirname(achados[0])
for a in ['train.xlsx', 'test1.xlsx', 'test2.xlsx', 'divisao_80_20.csv', 'folds_5.csv']:
    locais = glob.glob(f'/kaggle/input/**/{a}', recursive=True) + glob.glob(f'/content/**/{a}', recursive=True) + glob.glob(a)
    assert locais or os.path.exists(os.path.join(PASTA_DADOS, a)), f'{a} não está no Dataset: crie uma nova versão com ele'
    print('  ', a, '->', os.path.join(PASTA_DADOS, a) if os.path.exists(os.path.join(PASTA_DADOS, a)) else locais[0])
n_gpus = torch.cuda.device_count()
assert n_gpus > 0, 'sem GPU: ative o acelerador (GPU T4 x2)'
print(f'{n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}')

fila = [{'saida': RAIZ, 'pasta_dados': PASTA_DADOS, **COMUM, **e} for e in EXPERIMENTOS]
rodando = {}   # gpu -> (experimento, processo, arquivo de log)
concluidos = []


def iniciar(gpu, exp):
    pasta = os.path.join(RAIZ, exp['experimento'])
    os.makedirs(pasta, exist_ok=True)
    caminho_cfg = os.path.join(pasta, 'config_entrada.json')
    with open(caminho_cfg, 'w', encoding='utf-8') as f:
        json.dump(exp, f, indent=2, ensure_ascii=False)
    log = open(os.path.join(pasta, 'log.txt'), 'w', encoding='utf-8')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    proc = subprocess.Popen([sys.executable, 'ep2_treino.py', caminho_cfg], stdout=log, stderr=subprocess.STDOUT, env=env)
    rodando[gpu] = (exp['experimento'], proc, log)
    print(f'[GPU {gpu}] iniciado: {exp["experimento"]} ({exp.get("modelo")})', flush=True)


def ultima_linha(nome):
    try:
        with open(os.path.join(RAIZ, nome, 'log.txt'), encoding='utf-8', errors='replace') as f:
            linhas = [l.strip() for l in f if l.strip() and ('época' in l or 'epoca=' in l or '=====' in l or 'Error' in l)]
        return linhas[-1][:160] if linhas else '(carregando)'
    except FileNotFoundError:
        return ''


inicio = time.time()
while fila or rodando:
    for gpu in range(n_gpus):
        if gpu not in rodando and fila:
            iniciar(gpu, fila.pop(0))
    time.sleep(60)
    for gpu, (nome, proc, log) in list(rodando.items()):
        if proc.poll() is not None:
            log.close()
            status = 'ok' if proc.returncode == 0 else f'ERRO (código {proc.returncode}); veja {nome}/log.txt'
            concluidos.append((nome, status))
            print(f'[GPU {gpu}] terminou: {nome} -> {status}', flush=True)
            if proc.returncode != 0:
                with open(os.path.join(RAIZ, nome, 'log.txt'), encoding='utf-8', errors='replace') as f:
                    print('    | ' + '    | '.join(f.readlines()[-25:]), flush=True)
            del rodando[gpu]
    minutos = (time.time() - inicio) / 60
    print(f'--- {minutos:.0f} min | ' + ' || '.join(f'GPU {g} {n}: {ultima_linha(n)}' for g, (n, _, _) in rodando.items()),
          flush=True)

print('\\n===== Resumo =====')
for nome, status in concluidos:
    print(nome, '->', status)
    hist = os.path.join(RAIZ, nome, 'historico.csv')
    if os.path.exists(hist):
        print(pd.read_csv(hist).to_string(index=False))
''')

nb = {'cells': celulas, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
      'nbformat': 4, 'nbformat_minor': 5}
with open(os.path.join(PASTA, 'ep2_transformers.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('kaggle/ep2_transformers.ipynb gerado')
