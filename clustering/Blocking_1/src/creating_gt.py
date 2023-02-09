import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))
from itertools import combinations
from pathlib import Path
import pandas as pd
from Config import config
from tqdm.auto import tqdm
from utils.logger_helper import log_helper
from utils.trace import Trace


def create_gtruth(log):
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
        dtype={"postalCode": "str", "houseNumber": "str"},
    )

    Path(config.gt_dir).mkdir(parents=True, exist_ok=True)

    if config.is_inference == False:
        matches = set()
        for cluster_id, values in tqdm(df.groupby(["clusterId"])):
            names = values["Id"].tolist()
            comb = set([tuple(sorted(x)) for x in set(combinations(names, 2))])
            matches.update(comb)

        df_pairs = pd.DataFrame(matches)
        df_pairs.columns = ["ltable_id", "rtable_id"]
        df_pairs.drop_duplicates(inplace=True)
        df_pairs = df_pairs[~(df_pairs["ltable_id"] == df_pairs["rtable_id"])]
        df_pairs = pd.merge(
            df_pairs,
            df[["Id", "placeId"]],
            how="left",
            left_on=["ltable_id"],
            right_on=["Id"],
        )
        df_pairs = pd.merge(
            df_pairs,
            df[["Id", "placeId"]],
            how="left",
            left_on=["rtable_id"],
            right_on=["Id"],
            suffixes=[1, 2],
        )

        df_pairs = df_pairs[df_pairs["placeId1"] != df_pairs["placeId2"]]

        df_pairs = df_pairs[["ltable_id", "rtable_id", "placeId1", "placeId2"]]

        df_pairs.to_csv(config.gt_dir + f"GT_{config.country}.csv", index=None)
        log.info("Completed creating ground truth data")
        print("Completed creating ground truth data")


if __name__ == "__main__":
    trace = Trace()

    log = log_helper(f"create_gtruth_{config.country}", config.country)

    with trace.timer("creating_ground_truth", log):
        create_gtruth(log)
