import pandas as pd
import os

root = os.path.dirname(os.path.dirname(__file__))
for fname in ['dataset/fraudTrain.csv','dataset/fraudTest.csv']:
    path = os.path.join(root, fname)
    print('---', fname, '---')
    df = pd.read_csv(path)
    print('shape', df.shape)
    print('columns', list(df.columns))
    print('dtypes', df.dtypes.value_counts().to_dict())
    print('head', df.head(1).to_dict(orient='records'))
    print('target', df.columns[-1], df.iloc[:, -1].unique()[:10])
