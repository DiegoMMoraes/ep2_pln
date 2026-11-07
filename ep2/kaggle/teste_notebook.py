# Teste local do ep2_transformers.ipynb: roda as células do notebook (menos o %%writefile, pois ep2_treino.py já
# está nesta pasta) com dois experimentos mínimos (BERTimbau base e um Qwen3 minúsculo com LoRA), 300 textos.
# O Qwen3 minúsculo precisa existir em <scratch>/qwen_mini (ver o teste do caminho LoRA).
#
# Uso (de dentro de kaggle/): python teste_notebook.py <pasta_temporaria>

import json
import os
import sys

tmp = sys.argv[1]
nb = json.load(open('ep2_transformers.ipynb', encoding='utf-8'))
celulas = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
assert celulas[2].startswith('%%writefile ep2_treino.py')
assert celulas[2].split('\n', 1)[1].strip() == open('ep2_treino.py', encoding='utf-8').read().strip(), \
    'notebook desatualizado: rode gerar_notebook.py'

comum = dict(modos=['holdout'], epocas=1, amostra=300, pasta_dados=os.path.abspath('dados'))
config = f'''
EXPERIMENTOS = [
    dict(experimento='nb_encoder', modelo='neuralmind/bert-base-portuguese-cased', lr=2e-5),
    dict(experimento='nb_falha', modelo='modelo/que-nao-existe'),
    dict(experimento='nb_llm', modelo=r'{os.path.join(tmp, "qwen_mini")}', tipo='llm_lora', lr=1e-3,
         gradient_checkpointing=True),
]
COMUM = {comum!r}
'''
escopo = {}
for c in [config, celulas[1], celulas[3].replace("os.path.abspath('saida')", repr(os.path.join(tmp, 'saida_nb')))
                                .replace("glob.glob('train.xlsx')", "glob.glob('dados/train.xlsx')")]:
    exec(compile(c, 'notebook', 'exec'), escopo)
