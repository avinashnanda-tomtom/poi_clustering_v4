from models.gdf_category_mapping import gdf_map
from models.gdf_subcat_mapping import gdf_subcat_map
import numpy as np

def gdf_mapping(gdf):
    categ = []
    for i in gdf:
        if i in gdf_map:
            categ.append(gdf_map[i])

    return max(set(categ), key=categ.count)


def gdf_subcat_mapping(gdf):
    categ = []
    for i in gdf:
        if i in gdf_subcat_map:
            categ.append(gdf_subcat_map[i])

    if len(categ) == 0:
        return np.nan

    return max(set(categ), key=categ.count)


def map_categories(df):
    df["category"] = df["category"].str.lower()
    df["category"] = df["category"].apply(eval)
    df["subCategory"] = df["category"].apply(gdf_subcat_mapping)
    df["category"] = df["category"].apply(gdf_mapping)

    return df
