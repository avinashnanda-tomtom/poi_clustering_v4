# ---------------------------------- cosine and knn configs ---------------------------------- #

COSINE_NEIGHBORS = 30
cosine_model = "29880"
source_name_model = "paraphrase-multilingual-mpnet-base-v2"
category_model = "all-MiniLM-L6-v2"
country = "NLD"
non_english=False
# ------------------------------ pipeline config ----------------------------- #

is_inference = False

threshold_dict = {"NZL": 0.0001, "GLP": 0.0001, "ZAF": 0.0001, "AUS": 0.0001,"BEL":0.0001}

cols_to_block = [
    "sourceNames", "latitude", "longitude", "houseNumber", "streets", "cities",
    "brands", "postalCode", "category"
]

columns_keep = [
    "locality", "clusterId", "placeId", "officialName","sourceNames","subCategory", "category", "latitude",
    "longitude", "houseNumber", "streets", "cities", "postalCode", "Id",
    "brands", "phoneNumbers", "supplier","email","internet"
]

# -------------------------- directory configuration ------------------------- #
root_dir = "/workspace/clustering/"
model_dir = "/workspace/clustering/models/"
raw_dir = "/workspace/clustering/data/raw/"
input_dir = "/workspace/clustering/data/input/"
gt_dir = "/workspace/clustering/data/ground_truth/"
embed_dir = "/workspace/clustering/embeddings/"
output_stage1 = "/workspace/clustering/outputs/stage1/"
output_stage2 = "/workspace/clustering/outputs/stage2/"
final_result_dir = "/workspace/clustering/outputs/results/"

# ----------------------- model features and parameters ---------------------- #

lgb_level1_params = {
    "learning_rate": 0.05,
    "num_leaves": 500,
    "reg_alpha": 1,
    "reg_lambda": 10,
    "min_child_samples": 1000,
    "min_split_gain": 0.01,
    "min_child_weight": 0.01,
    "path_smooth": 0.1
}


All_columns = ['ltable_id', 'rtable_id','placeId1', 'placeId2', 'haversine',
       'name_davies', 'name_leven', 'name_dice', 'name_jaro', 'name_set_ratio',
       'street_davies', 'street_leven', 'street_jaro', 'email_davies',
       'email_leven', 'email_jaro', 'url_davies', 'url_leven', 'url_jaro',
       'brands_davies', 'brand_leven', 'brand_jaro', 'phone_lcs',
       'subcat_WRatio', 'subcat_ratio', 'subcat_token_set_ratio',
       'Is_direction_match_1', 'Is_direction_match_2', 'Is_house_match_0',
       'Is_house_match_1', 'Is_house_match_2', 'Is_category_match_0',
       'Is_category_match_1', 'Is_subcategory_match_0',
       'Is_subcategory_match_1', 'Is_subcategory_match_2', 'Is_brand_match_0',
       'Is_brand_match_1', 'Is_brand_match_2', 'Is_brand_match_3',
       'Is_related_cat_0', 'Is_related_cat_1', 'Is_name_number_match_2',
       'Is_name_number_match_3', 'is_phone_match_1', 'is_phone_match_2',
       'is_phone_match_3', 'Is_email_match_0', 'Is_email_match_1',
       'Is_email_match_2', 'Is_url_match_0', 'Is_url_match_1',
       'Is_url_match_2', 'country', 'Is_name_number_match_0',
       'Is_name_number_match_1', 'Is_direction_match_0', 'is_phone_match_4']


EM_features = ['similarity', 'haversine', 'name_davies',
       'name_leven', 'name_dice', 'name_jaro', 'name_set_ratio',
       'street_davies', 'street_leven', 'street_jaro', 'email_davies',
       'email_leven', 'email_jaro', 'url_davies', 'url_leven', 'url_jaro',
       'brands_davies', 'brand_leven', 'brand_jaro', 'phone_lcs',
       'subcat_WRatio', 'subcat_ratio', 'subcat_token_set_ratio',
       'Is_direction_match_0', 'Is_direction_match_1', 'Is_direction_match_2',
       'Is_house_match_0', 'Is_house_match_1', 'Is_house_match_2',
       'Is_category_match_0', 'Is_category_match_1', 'Is_subcategory_match_0',
       'Is_subcategory_match_1', 'Is_subcategory_match_2', 'Is_brand_match_0',
       'Is_brand_match_1', 'Is_brand_match_2', 'Is_brand_match_3',
       'Is_related_cat_0', 'Is_related_cat_1', 'Is_name_number_match_0',
       'Is_name_number_match_1', 'Is_name_number_match_2',
       'Is_name_number_match_3', 'is_phone_match_1', 'is_phone_match_2',
       'is_phone_match_3', 'is_phone_match_4', 'Is_email_match_0',
       'Is_email_match_1', 'Is_email_match_2', 'Is_url_match_0',
       'Is_url_match_1', 'Is_url_match_2']

class CFG:
    wandb = True
    apex = True
    print_freq = 400
    num_workers = 4
    out_dir = "/workspace/clustering/models/"
    batch_size = 64
    max_len = 128
    seed = 42
    text_cols = [
        'sourceNames', 'category', 'houseNumber', 'streets', 'cities','postalCode']
    numeric_cols = ['haversine', 'latdiff', 'manhattan', 'euclidean', 'all_similarities', 'name_similarities', 'category_similarities','cities_similarities', 'streets_similarities']
