import pandas as pd
from Config import config
from pathlib import Path
from utils.logger_helper import log_helper
from utils.trace import Trace
from tqdm.auto import tqdm
from itertools import combinations
from utils import blocking_utils


def add_exact_dups(log):
    df = pd.read_csv(config.final_result_dir + f"exact_duplicates_{config.country}.csv")
    df_clustered = pd.read_csv(
        f"{config.final_result_dir}/{config.country}_final_clustered.csv"
    )
    log.info(
        f"length of overall clustered before adding exact duplicates {len(df_clustered)}"
    )

    df["duplicate_placeId"] = df["duplicate_placeId"].apply(eval)
    duplicate_placeId = list(df["duplicate_placeId"])
    df_clustered_dict = dict(zip(df_clustered["placeId"], df_clustered["clusterId"]))

    clusterId = []
    id = []
    for id_list in duplicate_placeId:
        for i in id_list:
            if i in df_clustered_dict.keys():
                clusterId.append(df_clustered_dict[i])
                id.append(id_list)

    cl = []
    placeId = []
    for clust, pid in zip(clusterId, id):
        for i in pid:
            cl.append(clust)
            placeId.append(i)

    df_clusters_dups = pd.DataFrame(list(zip(placeId, cl)))
    df_clusters_dups.columns = ["placeId", "clusterId"]
    df_final_cluster = pd.concat([df_clustered, df_clusters_dups])
    df_final_cluster.drop_duplicates(inplace=True)

    log.info(
        f"length of overall clustered after adding exact duplicates {len(df_final_cluster)}"
    )

    if config.is_inference == False:
        df_raw_dups = pd.read_parquet(
            config.raw_dir + f"Fuse_{config.country}_with_dups.parquet"
        )
        matches = set()
        for _, values in tqdm(df_raw_dups.groupby(["clusterId"])):
            names = values["placeId"].tolist()
            comb = set([tuple(sorted(x)) for x in set(combinations(names, 2))])
            matches.update(comb)

        golden_df = pd.DataFrame(matches)
        golden_df.columns = ["placeId1", "placeId2"]
        golden_df.drop_duplicates(inplace=True)

        matches1 = set()
        for _, values in tqdm(df_final_cluster.groupby(["clusterId"])):
            names1 = values["placeId"].tolist()
            comb1 = set([tuple(sorted(x)) for x in set(combinations(names1, 2))])
            matches1.update(comb1)

        candidate_set_df = pd.DataFrame(matches1)
        candidate_set_df.columns = ["placeId1", "placeId2"]
        candidate_set_df.drop_duplicates(inplace=True)

        statistics_dict = blocking_utils.compute_blocking_statistics_placeid(
            candidate_set_df, golden_df
        )

        log.info(f"Metrics = {statistics_dict}")

    df_final_cluster.to_csv(
        f"{config.final_result_dir}/{config.country}_final_clustered_with_exact_dups.csv",
        index=None,
    )

    print("completed adding exact duplicates")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"add_exact_duplicates_{config.country}", config.country)
    with trace.timer("add_exact_duplicates", log):
        add_exact_dups(log)
