import glob
import dask.dataframe as dd
import pandas as pd
from Config import config
from tqdm.auto import tqdm
from utils.blocking_utils import chunk_size
from utils.logger_helper import log_helper
from utils.cleaning_utils import remove_path
from utils.trace import Trace
from utils.trace import reduce_mem_usage

def combine_feat(log):
    
    
    df_embeds = pd.DataFrame()
    for file in tqdm(
        glob.glob(
            config.root_dir
            + f"Entity_matching/generated_features_em/{config.country}_batch_embedding_features_em/*.parquet"
        )
    ):
        df = pd.read_parquet(file, engine="pyarrow")
        df_embeds = pd.concat([df_embeds, df], axis=0, ignore_index=True)
    df_embeds = reduce_mem_usage(df_embeds)
    log.info(f"shape of embedding feature dataframe = {df_embeds.shape}")
    
    
    df_distance = pd.DataFrame()

    for file in tqdm(
        glob.glob(
            config.root_dir
            + f"Entity_matching/generated_features_em/{config.country}_batch_dist_features_em/*.parquet"
        )
    ):
        df = pd.read_parquet(file, engine="pyarrow")
        df_distance = pd.concat([df_distance, df], axis=0, ignore_index=True)
        
    df_distance = reduce_mem_usage(df_distance)
    log.info(f"shape of distance feature dataframe = {df_distance.shape}")

    if df_distance.shape[0] != df_embeds.shape[0]:
        log.error(
            f"The embedding features shape  = {df_embeds.shape}  distance feature shape = {df_distance.shape} shape mismatch"
        )

    # log.info("remove distance and embedding features after combining")
    # remove_path(
    #     config.root_dir
    #     + f"Entity_matching/generated_features_em/{config.country}_batch_embedding_features_em"
    # )
    # remove_path(
    #     config.root_dir
    #     + f"Entity_matching/generated_features_em/{config.country}_batch_dist_features_em"
    # )

    df_features = pd.merge(
        df_embeds,
        df_distance,
        how="left",
        left_on=["ltable_id", "rtable_id"],
        right_on=["ltable_id", "rtable_id"],
    )
    


    log.info(f"The final shape of all features dataset =  {df_features.shape}")

    log.info("splitting the full dataframe into multiple chunks")
    
    # converting back to float32 as pyarrow does not support float16
    
    float16_cols = list(df_features.select_dtypes(include=['float16']).columns)
    new_type = dict((col,'float32') for col in float16_cols)
    df_features = df_features.astype(new_type)
    
    ddf = dd.from_pandas(df_features, chunksize=chunk_size(len(df_features)))

    ddf.to_parquet(
        config.root_dir
        + f"Entity_matching/generated_features_em/{config.country}_batch_all_features_em/",
        engine="pyarrow",
        compression="zstd",
        write_index=False,
        write_metadata_file=None,
    )

    print("Completed combining features")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Combining_all_features_em_{config.country}", config.country)

    with trace.timer("Combining_all_features", log):
        combine_feat(log)
