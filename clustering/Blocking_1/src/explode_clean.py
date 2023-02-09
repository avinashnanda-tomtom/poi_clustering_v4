import sys
from os.path import abspath, dirname
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(dirname(dirname(abspath(__file__))))
import numpy as np
import pandas as pd
from Config import config
from pathlib import Path
from utils.explode import explode_df
from utils.explode_multilingual import explode_df_multilingual
from utils.logger_helper import log_helper
from utils.mapping_categories import map_categories
from utils.cleaning_utils import canonical_url, clean_email, process_phone, clean_text, unique_list, clean_name, clean_streets, rem_words
from utils.trace import Trace
from tqdm.auto import tqdm
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4,verbose=1)


def explode_clean(log):
    df = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv",engine='c')
    Path(config.input_dir).mkdir(parents=True, exist_ok=True)
    
    log.info(f"length of overall dataframe {len(df)}")
    if config.non_english == True:
        df = explode_df_multilingual(df)
    else:
        df = explode_df(df)
    
    df = map_categories(df)
    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)
    df = df[~(df["longitude"].isnull())] 
    df["sourceNames"] = df["sourceNames"].str.strip()
    df["sourceNames"] = df["sourceNames"].replace("", np.nan)
    df = df[~(df["sourceNames"].isnull())]
    df.drop_duplicates(inplace=True)
    
    df["internet"] = df["internet"].str.strip("[]")
    df["email"] = df["email"].str.strip("[]")
    df["internet"] = df["internet"].apply(canonical_url)
    df["email"] = df["email"].apply(canonical_url)
    df["internet"] = df["internet"].apply(clean_email)
    df["email"] = df["email"].apply(clean_email)
    
    df["phoneNumbers"] = df["phoneNumbers"].apply(process_phone)

    print("Cleaning the data")
    text_columns = [
        "sourceNames", "brands", "houseNumber", "category", "streets",
        "cities", "postalCode", "subCategory", "email", "internet"
    ]

    for col in tqdm(text_columns):
        df[col] = df[col].astype("str").astype(str).replace('nan', np.nan)
        df[col] = df[col].apply(clean_text)


    # remove duplicate words
    df["brands"] = df["brands"].apply(unique_list)
    df["subCategory"] = df["subCategory"].apply(unique_list)
    df["sourceNames"] = df["sourceNames"].apply(unique_list)
    df["streets"] = df["streets"].apply(unique_list)
    df["cities"] = df["cities"].apply(unique_list)

    # remove unspecified sub category

    df["subCategory"].replace('unspecified', np.nan, regex=True, inplace=True)

    df["cities"] = df["cities"].apply(clean_streets)
    df["streets"] = df["streets"].apply(clean_streets)
    df["sourceNames"] = df["sourceNames"].apply(clean_name)

    df["sourceNames"] = df["sourceNames"].apply(rem_words)
    df["cities"] = df["cities"].apply(rem_words)
    df["streets"] = df["streets"].apply(rem_words)

    df = df.replace(r'^\s*$', np.nan, regex=True)

    df["sourceNames"] = df["sourceNames"].replace("", np.nan)
    df = df[~(df["sourceNames"].isnull())]

    log.info(f"length of after cleaning {len(df)}")
    df = df.sample(frac=1).reset_index(drop=True)
    df["Id"] = df.index
    
    for col in df.columns:
        if col not in config.columns_keep:
            df.drop(col, axis=1, inplace=True)

    df["category"].replace('caf pub', 'cafe pub', regex=True,inplace=True)

    log.info(f"length of after cleaning {len(df)}")

    df.to_csv(config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",
              index=False)
    print("completed cleaning")
    log.info("completed cleaning")


if __name__ == "__main__":
    trace = Trace()

    log = log_helper(f"explode_and_clean_{config.country}", config.country)

    with trace.timer("explode_and_clean", log):
        explode_clean(log)
