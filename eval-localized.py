import sys, os
import numpy as np
import pandas as pd
from config import cfg
from metric import eval

root_path = '/data/users/xux12/proj/fl-crosstrain/models'
store_dirs = ['model_20220128213838','model_20220128221531','model_20220128230401','model_20220128235337']
store_dirs = ['model_20220204222337','model_20220204230139','model_20220204235252','model_20220205004516']
store_dirs = ['model_20220205074027','model_20220205091709','model_20220205115611','model_20220205141534']
store_dirs = ['model_20220205210929','model_20220205224823','model_20220206012853','model_20220206034849']
store_dirs = ['model_20220206163903','model_20220206172106','model_20220206181621','model_20220206191221']
store_dirs = ['model_20220206213737','model_20220206222211','model_20220206231914','model_20220207001628']
store_dirs = ['model_20220207104410','model_20220207113347','model_20220207124337','model_20220207134855']
store_dirs = ['model_20220207161651','model_20220207171545','model_20220207185944','model_20220207202713']

metric_fn='metric_testing-global'
df_list = []
for store_dir in store_dirs:
    result_dir = '{}/{}/results_test'.format(root_path, store_dir)
    df = pd.read_csv('{}/metric_testing.csv'.format(result_dir))
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
df.to_csv('{}/{}/{}.csv'.format(root_path, store_dirs[0], metric_fn))

eval(pd_path='{}/{}'.format(root_path, store_dirs[0]), gt_entries=None, label_map=None, cls_num=cfg['cls_num'], metric_fn=metric_fn, calc_asd=True)