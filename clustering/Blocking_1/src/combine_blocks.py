import dask.dataframe as dd
import pandas as pd
from Config import config
from utils import blocking_utils
from utils.blocking_utils import chunk_size
from utils.logger_helper import log_helper
from utils.cleaning_utils import remove_path
from utils.trace import Trace
from utils.trace import reduce_mem_usage
from utils.create_features import add_lat_lon_distance_features
def combine(log):
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
        dtype={"postalCode": "str", "houseNumber": "str"},
    )
    log.info("Reading the candidates")


    candidates_cosine = pd.read_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_topk_cosine_{config.country}_{config.COSINE_NEIGHBORS}.parquet",
        columns=["ltable_id", "rtable_id", "placeId1", "placeId2"],
    )
    
    candidates_cosine = reduce_mem_usage(candidates_cosine)

    candidates_house = pd.read_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_house_number_{config.country}.parquet"
    )
    candidates_house = reduce_mem_usage(candidates_house)
    
    candidates_phone = pd.read_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_phone_number_{config.country}.parquet"
    )
    candidates_phone = reduce_mem_usage(candidates_phone)
    
    candidates_name = pd.read_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_source_name_{config.country}.parquet"
    )
    candidates_name = reduce_mem_usage(candidates_name)

    final_candidates = pd.concat(
        [candidates_cosine, candidates_house, candidates_phone, candidates_name]
    )


    
    del candidates_name,candidates_phone,candidates_cosine,candidates_house

    final_candidates.drop_duplicates(inplace=True)
    log.info(f"Total number of candidates after combining = {len(final_candidates)}")



    log.info("writing candidates to disk")
    final_candidates.reset_index(drop=True, inplace=True)

    log.info(f"Total number of candidates = {len(final_candidates)}")
    
    
    log.info(f"Total number of candidates = {len(final_candidates)}")
    print(f"Total number of candidates = {len(final_candidates)}")
    
    log.info(f"dropping candidates based on distance")
    print(f"dropping candidates based on distance")
    
    final_candidates = pd.merge(final_candidates, df[["Id","longitude","latitude"]],  how='left', left_on=[
                'ltable_id'], right_on=['Id'])
    final_candidates.drop('Id',inplace=True,axis=1)
    final_candidates = pd.merge(final_candidates, df[["Id","longitude","latitude"]],  how='left', left_on=[
                        'rtable_id'], right_on=['Id'],suffixes=["1","2"])

    final_candidates.drop('Id',inplace=True,axis=1)
    
    final_candidates = add_lat_lon_distance_features(final_candidates)
    
    
    
    km_dist = 1
    final_candidates = final_candidates[final_candidates["haversine"]<=km_dist]
    log.info(f"Total number of candidates after dropping above {km_dist} km = {len(final_candidates)}")
    print(f"Total number of candidates after dropping above {km_dist} km = {len(final_candidates)}")
    final_candidates = final_candidates[['ltable_id', 'rtable_id', 'placeId1', 'placeId2']]
    
    if config.is_inference == False:

        log.info("validating metrics with ground truth")

        golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")

        statistics_dict = blocking_utils.compute_blocking_statistics(
            final_candidates, golden_df, df
        )

        log.info(f"Metrics = {statistics_dict}")


    ddf = dd.from_pandas(final_candidates, chunksize=chunk_size(len(final_candidates)))

    ddf.to_parquet(
        config.output_stage1 + f"batch_candidates/{config.country}_parquet",
        engine="fastparquet",
        compression="zstd",
        write_index=False,
        write_metadata_file=None,
    )
    
    # remove_path(config.root_dir+ f"Blocking_1/outputs/")

    print("completed combining the candidates")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(
        f"combining_all_cos_{config.COSINE_NEIGHBORS}_{config.country}",
        config.country,
    )

    with trace.timer("combine_blocks", log):
        combine(log)
