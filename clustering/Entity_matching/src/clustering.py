from utils.trace import Trace
from utils.logger_helper import log_helper
from utils.create_cluster import agglomerative_clustering
from tqdm.auto import tqdm
from Config import config
import pandas as pd
from itertools import combinations
import uuid
from pathlib import Path
import sys
from os.path import abspath, dirname
from utils import blocking_utils
sys.path.append(dirname(dirname(abspath(__file__))))


def cluster(log):
    Path(f"{config.final_result_dir}").mkdir(parents=True, exist_ok=True)
    # df = pd.read_csv(
    #     config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
    #     dtype={"postalCode": "str", "houseNumber": "str"},
    # )
    df_prediction = pd.read_parquet(
        f"{config.final_result_dir}/{config.country}_results_final.parquet"
    )
    print("clustering")
    
    # placeId = df["placeId"].to_numpy()
    # df_prediction["placeId1"] = placeId[df_prediction['ltable_id'].to_numpy()]
    # df_prediction["placeId2"] = placeId[df_prediction['rtable_id'].to_numpy()]

    df_prediction = df_prediction.sort_values("prediction", ascending=False)

    if config.is_inference == False:
        golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")
        log.info("calculating metrics before cluster merging")

        df_pred = df_prediction[df_prediction["prediction"] > 0.5]

        statistics_dict = blocking_utils.compute_blocking_statistics_placeid(df_pred, golden_df)

        log.info(f"Metrics = {statistics_dict}")

    log.info("Merging clusters for final output")

    df_p = df_prediction.drop_duplicates(["placeId1", "placeId2"], keep="first")
    df_final = df_p[["placeId1", "placeId2"]].drop_duplicates()
    df_final.columns = ["id_1", "id_2"]
    df_final["id_1"] = df_final["id_1"].map(str)
    df_final["id_2"] = df_final["id_2"].map(str)
    preds = agglomerative_clustering(df_final, list(df_p["prediction"]))
    preds.columns = ["placeId", "clusterId"]

    log.info("Creating uuid for clusters")

    grouped = preds.groupby("clusterId")
    placeId = []
    clusterId = []
    for _, df_group in grouped:
        cl = str(uuid.uuid4())
        for _, row in df_group.iterrows():
            placeId.append(row["placeId"])
            clusterId.append(cl)

    df_clusters = pd.DataFrame(list(zip(placeId, clusterId)))
    df_clusters.columns = ["placeId", "clusterId"]

    log.info("adding pois which are not part of cluster as single cluster poi")

    df_raw = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv")

    single_placeId = []
    single_clusterId = []
    dict_poi_c = dict(zip(df_clusters["placeId"], df_clusters["placeId"]))
    for poid in tqdm(set(list(df_raw["placeId"]))):
        if poid not in dict_poi_c:
            single_placeId.append(poid)
            single_clusterId.append(str(uuid.uuid4()))

    df_clusters_single = pd.DataFrame(list(zip(single_placeId, single_clusterId)))
    df_clusters_single.columns = ["placeId", "clusterId"]

    df_final_cluster = pd.concat([df_clusters, df_clusters_single])

    log.info(f"Final shape of clustered {df_final_cluster.shape}")

    df_final_cluster.to_csv(
        f"{config.final_result_dir}/{config.country}_final_clustered.csv", index=None
    )

    if config.is_inference == False:
        log.info("calculating metrics after cluster merging")
        matches = set()
        for _, values in tqdm(df_clusters.groupby(["clusterId"])):
            names = values["placeId"].tolist()
            comb = set([tuple(sorted(x)) for x in set(combinations(names, 2))])
            matches.update(comb)
        df_pairs = pd.DataFrame(matches)
        df_pairs.columns = ["placeId1", "placeId2"]

        gold = []
        for _, row in golden_df.iterrows():
            gold.append(sorted([row["placeId1"], row["placeId2"]]))
        gold = pd.DataFrame(gold)
        gold.columns = ["placeId1", "placeId2"]

        statistics_dict = blocking_utils.compute_blocking_statistics_placeid(df_pairs, gold)

        log.info(f"final Metrics = {statistics_dict}")

    print("Completed clustering")
    


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Final_clustering_{config.country}", config.country)

    with trace.timer("Final Clustering", log):
        cluster(log)
