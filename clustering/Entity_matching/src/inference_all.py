import pandas as pd
import numpy as np
from utils.model_infer import pred_multi, pred_multi_xgb, pred_multi_catboost
from utils.metrics import print_metrics
import glob
from tqdm import tqdm
from pathlib import Path
from Config import config
from utils.trace import reduce_mem_usage
from utils.logger_helper import log_helper
from utils.trace import Trace
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def inference_all_models(log):
    Path(config.final_result_dir).mkdir(parents=True, exist_ok=True)
    file_list = glob.glob(
        config.root_dir +
        f"Entity_matching/generated_features_em/{config.country}_batch_all_features_em/*.parquet"
    )

    lgb_models = glob.glob(
        f"/workspace/clustering/Entity_matching/models/model_duplicate_gsplit_lgb*"
    )

    df_pairs = pd.DataFrame()
    for file in file_list:
        df1 = pd.read_parquet(file, engine='pyarrow')
        df_pairs = pd.concat([df_pairs, df1])

    df_pairs = reduce_mem_usage(df_pairs)

    # Light gbm model prediction

    all_pred = []
    for model_file in tqdm(lgb_models):
        prediction = pred_multi(model_file, df_pairs[config.EM_features])
        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    pred = np.mean(all_pred, axis=0)
    df_pairs["prediction_lgb_probab"] = pred

    # Xgboost model prediction

    print("predicting using xgboost")
    xgb_models = glob.glob(
        f"/workspace/clustering/Entity_matching/models/xgboost_dedup_v2_*")

    all_pred = []
    for model_file in tqdm(xgb_models):
        prediction = pred_multi_xgb(model_file, df_pairs[config.EM_features])
        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    pred = np.mean(all_pred, axis=0)
    df_pairs["prediction_xgb_probab"] = pred

    # catboost model prediction
    print("predicting using catboost")

    cat_models = glob.glob(
        f"/workspace/clustering/Entity_matching/models/catboost_dedup_*")

    all_pred = []
    for model_file in tqdm(cat_models):
        prediction = pred_multi_catboost(model_file,
                                         df_pairs[config.EM_features])
        all_pred.append(prediction)

    all_pred = np.array(all_pred)
    pred = np.mean(all_pred, axis=0)
    df_pairs["prediction_catboost_probab"] = pred

    # ensembling prediction

    df_pairs["prediction"] = df_pairs["prediction_lgb_probab"] + df_pairs[
        "prediction_xgb_probab"] + df_pairs["prediction_catboost_probab"]

    df_pairs["prediction"] = df_pairs["prediction"] / 3

    cols = [
        'ltable_id', 'rtable_id', 'placeId1', 'placeId2',
        'prediction_lgb_probab', 'prediction_xgb_probab',
        'prediction_catboost_probab', 'prediction', 'duplicate_flag'
    ]

    if config.is_inference == False:

        cols = [
            'ltable_id', 'rtable_id', 'placeId1', 'placeId2',
            'prediction_lgb_probab', 'prediction_xgb_probab',
            'prediction_catboost_probab', 'prediction', 'duplicate_flag'
        ]

        pred_1 = (np.array(df_pairs["prediction"]) > 0.5) * 1
        duplicate_flag = np.array(df_pairs["duplicate_flag"])
        log.info(
            f"The precision_score is {precision_score(duplicate_flag, pred_1)}"
        )
        log.info(f"The recall_score is {recall_score(duplicate_flag, pred_1)}")
        log.info(
            f"The accuracy_score is {accuracy_score(duplicate_flag, pred_1)}")
        log.info(f"The f1_score is {f1_score(duplicate_flag, pred_1)}")
        log.info(
            f"The confusion_matrix is {confusion_matrix(duplicate_flag, pred_1)}"
        )

    float16_cols = list(df_pairs.select_dtypes(include=['float16']).columns)
    new_type = dict((col, 'float32') for col in float16_cols)
    df_pairs = df_pairs.astype(new_type)

    df_pairs[cols].to_parquet(
        f"{config.final_result_dir}/{config.country}_results_final.parquet",
        engine="pyarrow",
        compression="zstd",
        index=None,
    )

    log.info("completed generating prediction")

    print("completed generating prediction")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Inference_stage2_em_{config.country}", config.country)

    with trace.timer("Forest_inference_em", log):
        inference_all_models(log)
