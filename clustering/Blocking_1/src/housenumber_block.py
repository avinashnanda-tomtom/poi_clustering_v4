import pandas as pd
from Config import config
from utils import blocking_utils
from utils.logger_helper import log_helper
from utils.trace import Trace
from tqdm.auto import tqdm
import numpy as np


def house_block(log):
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
        dtype={"postalCode": "str", "houseNumber": "str"},
    )

    log.info("Finding nearest neigbhours using housenumber")
    print(df.shape)

    p3 = df[["Id", "houseNumber", "latitude", "longitude"]].copy()

    p3 = p3[~(p3["houseNumber"].isnull())]
    print(p3.shape)


    # rounded coordinates
    # 1 degree latitude is approx 111 km
    # .1 degree is 11.1km
    # rounding: 1=11.1Km, 2=1.11Km

    p3["latitude"] = np.round(p3["latitude"], 2).astype("float32")
    p3["longitude"] = np.round(p3["longitude"], 2).astype("float32")

    p3 = p3.sort_values(by=["latitude", "longitude", "houseNumber"]).reset_index(
        drop=True
    )

    idx1 = []
    idx2 = []
    house = p3["houseNumber"].to_numpy()
    lon2 = p3["longitude"].to_numpy()
    for i in tqdm(range(p3.shape[0] - 1)):
        li = lon2[i]
        for j in range(1, min(1000, p3.shape[0] - 1 - i)):
            if li != lon2[i + j]:  # if lon and lat match - b/c of sorting order
                break
            if house[i] == house[i + j]:
                idx1.append(i)
                idx2.append(i + j)

    p1 = p3[["Id"]].loc[idx1].reset_index(drop=True)
    p2 = p3[["Id"]].loc[idx2].reset_index(drop=True)
    pairs_house_number = pd.DataFrame(zip(list(p1["Id"]), list(p2["Id"])))
    pairs_house_number.columns = ["Id1", "Id2"]
    pairs_house_number.drop_duplicates(inplace=True)
    pairs_house_number = pairs_house_number.reset_index(drop=True)
    # flip - only keep one of the flipped pairs
    idx = pairs_house_number["Id1"] > pairs_house_number["Id2"]
    pairs_house_number["t"] = pairs_house_number["Id1"]
    pairs_house_number["Id1"].loc[idx] = pairs_house_number["Id2"].loc[idx]
    pairs_house_number["Id2"].loc[idx] = pairs_house_number["t"].loc[idx]
    pairs_house_number = pairs_house_number[["Id1", "Id2"]]
    pairs_house_number = pairs_house_number.drop_duplicates()
    pairs_house_number = pairs_house_number[["Id1", "Id2"]]
    pairs_house_number.columns = ["ltable_id", "rtable_id"]

    candidate_set_df = blocking_utils.clean_placeid(pairs_house_number, df)

    if config.is_inference == False:

        log.info("validating metrics with ground truth")

        golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")

        statistics_dict = blocking_utils.compute_blocking_statistics(
            candidate_set_df, golden_df, df
        )

        log.info(f"Metrics = {statistics_dict}")

    log.info("writing candidates to disk")
    
    candidate_set_df.reset_index(drop=True, inplace=True)

    log.info(f"Total number of candidates = {len(candidate_set_df)}")

    candidate_set_df.to_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_house_number_{config.country}.parquet",
        compression="zstd",
        index=None,
    )


if __name__ == "__main__":

    trace = Trace()
    log = log_helper(
        f"Housenumber_neighours_{config.country}",
        config.country,
    )

    with trace.timer("generate_house_candidates", log):
        house_block(log)
