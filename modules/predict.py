import dill
import pandas as pd
import os
import json
from datetime import datetime


def predict():
    path = os.environ.get('PROJECT_PATH', '.')

    with open(f'{path}/data/models/cars_pipe_202206211454.pkl', 'rb') as dillfile:
        model = dill.load(dillfile)
    fds = sorted(os.listdir(f'{path}/data/test/'))
    dict_result = dict()
    for test_file in fds:
        with open(f'{path}/data/test/{test_file}') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            df.loc[:, 'age_category'] = df['year'].apply(
                lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
            df = df.copy()

            def short_model(x):
                if not pd.isna(x):
                    return x.lower().split(' ')[0]
                else:
                    return x

            df.loc[:, 'short_model'] = df['model'].apply(short_model)
            predmodel = model.predict(df)
            dict_result[form["id"]] = predmodel[0]
            df = pd.DataFrame.from_dict(dict_result, orient='index')
            df.to_csv(f'{path}/data/predictions/predict{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
