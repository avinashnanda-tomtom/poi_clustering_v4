import pandas as pd
import json

df_cat = pd.read_csv("/workspace/category_transformations-EU.csv")

df_cat["internal_category_description"] =  df_cat["internal_category_description"].str.lower()

df_cat_gdf = df_cat[~df_cat["gdf_category_description"].isnull()]
gdf_map = df_cat_gdf[["internal_category_description","gdf_category_description"]]
gdf_map.drop_duplicates(inplace=True)
gdf_map = dict(zip(gdf_map["internal_category_description"],gdf_map["gdf_category_description"]))

df_cat_subcat = df_cat[~df_cat["gdf_sub_cat_description"].isnull()]
gdf_subcat_map = df_cat_subcat[["internal_category_description","gdf_sub_cat_description"]]
gdf_subcat_map.drop_duplicates(inplace=True)
gdf_subcat_map = dict(zip(gdf_subcat_map["internal_category_description"],gdf_subcat_map["gdf_sub_cat_description"]))



# create json object from dictionary
json1 = json.dumps(gdf_map)
# open file for writing, "w" 
f = open("/workspace/clustering/models/gdf_map.json","w")
# write json object to file
f.write(json1)
# close file
f.close()

# create json object from dictionary
json2 = json.dumps(gdf_subcat_map)
# open file for writing, "w" 
f = open("/workspace/clustering/models/gdf_subcat_mapping.json","w")
# write json object to file
f.write(json2)
# close file
f.close()

# Format and rename file to .py manually

