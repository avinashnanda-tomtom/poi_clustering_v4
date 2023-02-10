import pandas as pd
from Config import config
from utils import blocking_utils
from utils.logger_helper import log_helper
from utils.trace import Trace
from tqdm.auto import tqdm
import numpy as np
import rapidfuzz
from itertools import compress
from utils.cleaning_utils import clean_text

def phone_block(log):
    df = pd.read_csv(config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',dtype={"postalCode": "str", "houseNumber": "str"})
    log.info("Finding nearest neigbhours using phonenumber")
    df1 = df.copy()
    print(df1.shape)

    df1 = df1[~(df1["phoneNumbers"].isnull())]
    df1["phoneNumbers"] = df1["phoneNumbers"].apply(clean_text)
    print(df1.shape)

    p3 = df1[["Id", "phoneNumbers", "latitude", "longitude"]].copy()

    p3["latitude"] = np.round(p3["latitude"], 1).astype("float32")
    p3["longitude"] = np.round(p3["longitude"], 2).astype("float32")

    p3 = p3.sort_values(by=["latitude", "longitude", "phoneNumbers"]).reset_index(
        drop=True
    )

    idx1 = []
    idx2 = []
    phone = p3["phoneNumbers"].to_numpy()
    lon2 = p3["longitude"].to_numpy()
    for i in tqdm(range(p3.shape[0] - 1)):
        li = lon2[i]
        selected_li = [j for j in range(i+1,min(i+500, p3.shape[0] - 1)) if lon2[j]== li]
        if len(selected_li)>=1:
            values = rapidfuzz.process.cdist(list([phone[i]]), list(phone[selected_li]), scorer=rapidfuzz.distance.LCSseq.similarity, score_cutoff=10)
            sel_idx = list(compress(selected_li, values[0] >=10))
            for j in sel_idx:
                idx1.append(i)
                idx2.append(j)
                
                

    p1 = p3[["Id"]].loc[idx1].reset_index(drop=True)
    p2 = p3[["Id"]].loc[idx2].reset_index(drop=True)
    pairs_phone_number = pd.DataFrame(zip(list(p1["Id"]), list(p2["Id"])))
    pairs_phone_number.columns = ["Id1", "Id2"]
    pairs_phone_number.drop_duplicates(inplace=True)
    pairs_phone_number = pairs_phone_number.reset_index(drop=True)
    # flip - only keep one of the flipped pairs
    idx = pairs_phone_number["Id1"] > pairs_phone_number["Id2"]
    pairs_phone_number["t"] = pairs_phone_number["Id1"]
    pairs_phone_number["Id1"].loc[idx] = pairs_phone_number["Id2"].loc[idx]
    pairs_phone_number["Id2"].loc[idx] = pairs_phone_number["t"].loc[idx]
    pairs_phone_number = pairs_phone_number[["Id1", "Id2"]]
    pairs_phone_number = pairs_phone_number.drop_duplicates()
    pairs_phone_number = pairs_phone_number[["Id1", "Id2"]]
    pairs_phone_number.columns = ["ltable_id", "rtable_id"]

    candidate_set_df = blocking_utils.clean_placeid(pairs_phone_number, df)


    candidate_set_df.reset_index(drop=True, inplace=True)

    log.info(f"Total number of candidates = {len(candidate_set_df)}")

    candidate_set_df.to_parquet(
        config.root_dir
        + f"Blocking_1/outputs/candidates_phone_number_{config.country}.parquet",
        compression="zstd",
        engine="pyarrow",
        index=None,
    )


if __name__ == "__main__":

    trace = Trace()
    log = log_helper(
        f"Phonenumber_neighours_{config.country}",
        config.country,
    )

    with trace.timer("generate_phone_candidates", log):
        phone_block(log)
