import gc

import dask.dataframe as dd
import joblib
import pandas as pd
from Config import config
from utils import blocking_utils
from pathlib import Path
from utils.logger_helper import log_helper
from utils.trace import Trace

from clustering.Blocking_1.src.cosine_neighours import cos_candidates


def gen_cos_candidates(log):
    df = pd.read_csv(config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',dtype={"postalCode": "str", "houseNumber": "str"})
    Path(config.root_dir + "Blocking_1/outputs").mkdir(parents=True, exist_ok=True)
    log.info("Loading embeddings")
    embeddings = joblib.load(config.embed_dir + f"embeddings_{config.country}.pkl")
    log.info("Searching for top k candidates")
    similarities, topK_indices_each_row = cos_candidates(
        df, embeddings, K=config.COSINE_NEIGHBORS, embed_dim=embeddings.shape[1]
    )
    del embeddings
    gc.collect()

    log.info("create combinations of the top k")
    

    all_combination = blocking_utils.create_pairs_cosine(
        df, topK_indices_each_row, similarities
    )

    candidate_set_df = pd.DataFrame(all_combination)
    # deleting all_combination to free space
    del all_combination
    gc.collect()


    candidate_set_df.columns = ["ltable_id", "rtable_id", "similarity"]
    candidate_set_df = blocking_utils.clean_placeid(candidate_set_df, df)
        
    if config.is_inference == False:

        log.info("validating metrics with ground truth")

        golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")

        statistics_dict = blocking_utils.compute_blocking_statistics(
            candidate_set_df, golden_df, df
        )

        log.info(f"Metrics = {statistics_dict}")

    log.info("writing candidates to disk")

    log.info(f"Total number of candidates = {len(candidate_set_df)}")

    # candidate_set_df.drop(["Id1", "Id2"], axis=1, inplace=True)

    candidate_set_df.reset_index(drop=True, inplace=True)


    candidate_set_df.to_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_topk_cosine_{config.country}_{config.COSINE_NEIGHBORS}.parquet",
        compression="zstd",
        engine="pyarrow",
        index=None,
    )

    print("Completed generating cosine candidates")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(
        f"COSINE_ANN_{config.country}_{config.COSINE_NEIGHBORS}", config.country
    )

    with trace.timer("generate_cosine_candidates", log):
        gen_cos_candidates(log)
