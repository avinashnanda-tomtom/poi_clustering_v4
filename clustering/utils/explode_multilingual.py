import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("once")
pd.set_option("display.max_columns", None)
from utils.cleaning_utils import combine_category, convert_to_List_multilingual,latin_select
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4,verbose=1)
from Config import config

String_columns = [
    "locality","placeId","sourceNames","houseNumber",
    "streets","cities","category","postalCode"
]


def explode_df_multilingual(df):
    """Function to explode the poi data.

    Args:
        df (dataframe): Dataframe to explode.

    Returns:
        dataframe: Exploded dataframe
    """
    print(f"length of overall dataframe {len(df)}")

    df = convert_to_List_multilingual(df)
    
    
    df["sourceNames"] = df.apply(lambda x: latin_select(x.sourceNames, x.script_names), axis=1)
    df["cities"] = df.apply(lambda x: latin_select(x.cities, x.script_city), axis=1)
    df["streets"] = df.apply(lambda x: latin_select(x.streets, x.script_city), axis=1)
    df["latitude"] = df.apply(lambda x: latin_select(x.latitude, x.script_city), axis=1)
    df["longitude"] = df.apply(lambda x: latin_select(x.longitude, x.script_city), axis=1)
    df["houseNumber"] = df.apply(lambda x: latin_select(x.houseNumber, x.script_city), axis=1)
    df["postalCode"] = df.apply(lambda x: latin_select(x.postalCode, x.script_city), axis=1)
        
        

    df = df.explode(["sourceNames"])

    print(f"length of overall dataframe after exploding sourceNames is {len(df)}")

    df["latitude_len"] = df["latitude"].apply(lambda x: len(x))
    df["longitude_len"] = df["longitude"].apply(lambda x: len(x))
    df["houseNumber_len"] = df["houseNumber"].apply(lambda x: len(x))
    df["streets_len"] = df["streets"].apply(lambda x: len(x))
    df["cities_len"] = df["cities"].apply(lambda x: len(x))
    df["postalCode_len"] = df["postalCode"].apply(lambda x: len(x))

    df["to_drop"] = np.where(
        (df["streets_len"] != df["latitude_len"])
        | (df["houseNumber_len"] != df["latitude_len"])
        | (df["cities_len"] != df["latitude_len"])
        | (df["postalCode_len"] != df["latitude_len"]),
        float("nan"),
        1.0,
    )

    df = df[~(df["to_drop"].isnull())]

    print(f"length of overall dataframe non matching list columns {len(df)}")

    df = df.explode(
        ["latitude", "longitude", "houseNumber", "streets", "cities", "postalCode"]
    )

    df.drop(
        [
            "streets_len",
            "latitude_len",
            "houseNumber_len",
            "longitude_len",
            "cities_len",
            "postalCode_len",
            "to_drop",
            "script_city",
            "script_names",
            "script_streets"
        ],
        axis=1,
        inplace=True,
    )

    print(f"length of overall dataframe after exploding all columns is {len(df)}")

    df["category"] = df.apply(
        lambda x: combine_category(
            x.rawCategories, x.insertedCategories, x.preemptiveCategories
        ),
        axis=1,
    )

    df["category"] = df["category"].apply(
        lambda x: [i for i in x if i != "unspecified"]
    )

    df["brands"] = df["brands"].apply(lambda x: list(set(x)))

    df.drop(
        ["rawCategories", "insertedCategories", "preemptiveCategories"],
        axis=1,
        inplace=True,
    )

    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)

    df = df[~((df["latitude"] == 0) & (df["longitude"] == 0))]

    print(f"length after dropping where lat and long both are zero {len(df)}")
    df["brands"] = df["brands"].apply(lambda x: " ".join(sorted(set(x), key=x.index)))

    df["latitude"] = np.round(df["latitude"], 5).astype(np.float32)
    df["longitude"] = np.round(df["longitude"], 5).astype(np.float32)

    # dropping where source name is null
    df["sourceNames"] = df["sourceNames"].replace("", np.nan)

    df = df[~(df["sourceNames"].isnull())]

    print(f"length after dropping null sourcenames {len(df)}")

    print(f"length of final explode {len(df)}")

    return df
