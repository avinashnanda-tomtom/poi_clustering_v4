import pandas as pd
from Config import config
from pathlib import Path
from utils.logger_helper import log_helper
from utils.trace import Trace
from utils.cleaning_utils import drop_zero_coord
from datetime import datetime
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4,verbose=1)

def remove_exact_dups(log):
    
    Path(config.final_result_dir).mkdir(parents=True, exist_ok=True)
    date_log = datetime.now().strftime("%d_%m_%Y")
    
    df = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv",engine='c')
    df.to_parquet(config.raw_dir + f"Fuse_{config.country}_archive_{date_log}.parquet",engine='pyarrow',compression="zstd",index=None)

    df = drop_zero_coord(df)
    df = df.reset_index(drop=True)

    log.info(f"length of overall dataframe after dropping zero coordinates {len(df)}")

    df.to_parquet(config.raw_dir + f"Fuse_{config.country}_with_dups.parquet",engine='pyarrow',compression="zstd",index=None)
    log.info(f"length of overall dataframe before dropping exact duplicates {len(df)}")


    df1 = df.copy()

    cols = ["supplier","sourceNames","rawCategories","insertedCategories",
    "preemptiveCategories","brands","phoneNumbers","internet","email","latitude",
    "longitude","houseNumber","streets","cities","postalCode"]


    df1["all"] = df1[cols].astype(str).agg(" ".join, axis=1)

    df_duplicates = (
        df1.groupby('all').agg({'placeId':pd.Series.to_list}).reset_index()
    )
    df_duplicates.columns = ["all","duplicate_placeId"]

    df_duplicates["new_len"] = df_duplicates["duplicate_placeId"].apply(len)

    max_dup = df_duplicates["new_len"].max()

    log.info(f"maximum number of duplicates per group {max_dup}")

    log.info(f"saving all exact duplicates to be added later")

    df_duplicates[df_duplicates["new_len"] > 1].to_csv(
        config.final_result_dir + f"exact_duplicates_{config.country}.csv", index=None
    )

    df.drop_duplicates(subset=cols, inplace=True)

    log.info(f"length of overall dataframe after dropping exact duplicates {len(df)}")

    df.to_csv(config.raw_dir + f"Fuse_{config.country}.csv",index=None)

    log.info(f"total duplicates dropped {len(df1)-len(df)}")

    print("completed dropping exact duplicates")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"drop_exact_duplicates_{config.country}", config.country)

    with trace.timer("drop_exact_duplicates", log):
        remove_exact_dups(log)
