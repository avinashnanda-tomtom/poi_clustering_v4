import gc
import glob
from pathlib import Path
import ray
import numpy as np
import joblib
import pandas as pd

# import ray.util.multiprocessing as ray
from Config import config
from tqdm.auto import tqdm
from utils.create_features import fast_cosine
from utils.logger_helper import log_helper
from utils.trace import Trace
import psutil


@ray.remote
def find_embeddings_cosine(embed_all, filename):
    df = pd.read_parquet(filename, engine="pyarrow")
    similarity = list(
        fast_cosine(
            embed_all[list(df["ltable_id"])],
            embed_all[list(df["rtable_id"])],
            is_batch=True,
            batch=10000,
        )
    )
    df_embedding_features = pd.DataFrame(
        list(zip(list(df["ltable_id"]), list(df["rtable_id"]), similarity))
    )
    df_embedding_features.columns = ["ltable_id", "rtable_id", "similarity"]
    return df_embedding_features


def parallel_embedding_features(log):
    ray.shutdown()
    file_list = glob.glob(
        config.output_stage1 + f"batch_candidates/{config.country}_parquet/*.parquet"
    )
    Path(
        config.root_dir
        + f"Entity_matching/generated_features_em/{config.country}_batch_embedding_features_em"
    ).mkdir(parents=True, exist_ok=True)
    log.info("Loading the embedding pickles")
    embed_all = joblib.load(config.embed_dir + f"embeddings_name_{config.country}.pkl")
    memory = int(psutil.virtual_memory().total / 2.0**30)

    if memory >= 64:
        num_cpus = 1
    else:
        num_cpus = 1
    log.info(f"Total memory = {memory} number of parallel process = {num_cpus}")
    ray.init(num_cpus=num_cpus)
    obj_ref = ray.put(embed_all)
    del embed_all
    gc.collect()
    futures = []
    count = 0
    for i in tqdm(range(len(file_list))):
        count += 1
        futures.append(find_embeddings_cosine.remote(obj_ref, file_list[i]))
        if (count % num_cpus == 0) or (count == len(file_list)):
            results = ray.get(futures)
            df_embedding_features = pd.concat(results, axis=0, ignore_index=True)
            futures = []
            df_embedding_features.to_parquet(
                config.root_dir
                + f"Entity_matching/generated_features_em/{config.country}_batch_embedding_features_em/parquet_embedding_{i}.parquet",
                engine="pyarrow",
                compression="zstd",
                index=None,
            )
    ray.shutdown()
    log.info("completed generating embedding features")
    print("completed generating embedding features")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"generate_embedding_features{config.country}", config.country)
    with trace.timer("generate_embedding_features", log):
        parallel_embedding_features(log)
