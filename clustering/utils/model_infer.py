import lightgbm as lgbm
import numpy as np
import torch
from tqdm.auto import tqdm
import gc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def ForestInfer_lgb(ml_model, X, model_file, batch_size=128):
    """_summary_

    Args:
        ml_model (model): ForestInference model
        X (array): array of dataframe to predict on.
        model_file (text): ligtgbm pretrained model filename.
        batch_size (int, optional): batch size of prediction. Defaults to 128.

    Returns:
        array: prediction probabilities.
    """
    if torch.cuda.is_available():
        step = X.shape[0] // batch_size
        if batch_size * step < X.shape[0]:
            step += 1
        ret = []
        start = 0
        for i in tqdm(range(step)):
            end = start + batch_size

            pred = ml_model.predict(X[start:end, :])
            ret.append(pred)
            start += batch_size
        pred = np.concatenate(ret)

    else:
        model = lgbm.Booster(model_file=model_file)
        pred = model.pred(X)
    return pred


def ForestInfer_lgb(ml_model, X, model_file, batch_size=128):
    """_summary_

    Args:
        ml_model (model): ForestInference model
        X (array): array of dataframe to predict on.
        model_file (text): ligtgbm pretrained model filename.
        batch_size (int, optional): batch size of prediction. Defaults to 128.

    Returns:
        array: prediction probabilities.
    """
    if torch.cuda.is_available():
        step = X.shape[0] // batch_size
        if batch_size * step < X.shape[0]:
            step += 1
        ret = []
        start = 0
        for i in tqdm(range(step)):
            end = start + batch_size

            pred = ml_model.predict(X[start:end, :])
            ret.append(pred)
            start += batch_size
        pred = np.concatenate(ret)

    else:
        model = lgbm.Booster(model_file=model_file)
        pred = model.pred(X)
    return pred


def pred_multi(model_file,df):
    from cuml import ForestInference
    fi = ForestInference(output_type='numpy')
    ml_model = fi.load(filename=model_file, model_type='lightgbm')
    prediction = []
    pred = ForestInfer_lgb(ml_model,df.values,model_file,batch_size = 1024)
    prediction = prediction + list(pred)

    del ml_model
    gc.collect()
    torch.cuda.empty_cache()
    return prediction


def pred_multi_xgb(model_file,df,batch_size=200000):
    xgb_model = XGBClassifier()
    xgb_model.load_model(model_file)
    xgb_model.set_params(predictor="gpu_predictor")
    step = df.shape[0] // batch_size
    if batch_size * step < df.shape[0]:
        step += 1
        
    ret = []
    start = 0
    for i in tqdm(range(step)):
        end = start + batch_size

        pred = xgb_model.predict_proba(df[start:end])[:, 1]
        ret.append(pred)
        start += batch_size
    pred = np.concatenate(ret)
    del xgb_model
    gc.collect()
    return pred

def pred_multi_catboost(model_file,df,batch_size=200000):
    cat_model = CatBoostClassifier(task_type='GPU')
    cat_model.load_model(model_file)
    step = df.shape[0] // batch_size
    if batch_size * step < df.shape[0]:
        step += 1
        
    ret = []
    start = 0
    for i in tqdm(range(step)):
        end = start + batch_size

        pred = cat_model.predict_proba(df[start:end])[:, 1]
        ret.append(pred)
        start += batch_size
    pred = np.concatenate(ret)
    del cat_model
    gc.collect()
    return pred

# def pred_multi_catboost(model_file,df):
#     cat_model = CatBoostClassifier()
#     cat_model.load_model(model_file)
#     prediction = cat_model.predict_proba(df)[:, 1]
#     del cat_model
#     gc.collect()
#     return prediction