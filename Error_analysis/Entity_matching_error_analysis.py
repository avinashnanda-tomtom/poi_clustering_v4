import pandas as pd
from Config import config
from tqdm.auto import tqdm
from itertools import combinations
from datetime import datetime
if __name__ == "__main__":
    date_log = datetime.now().strftime("%d_%m_%Y")
    golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")
    df_raw = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv",engine='c')
    candidate_set_df = pd.read_csv(
        f"/workspace/clustering/outputs/results/{config.country}_final_clustered.csv"
    )

    matches = set()
    for cluster_id, values in tqdm(candidate_set_df.groupby(["clusterId"])):
        names = values["placeId"].tolist()
        comb = set([tuple(sorted(x)) for x in set(combinations(names, 2))])
        matches.update(comb)
    df_pairs = pd.DataFrame(matches)
    df_pairs.columns = ['placeId1', 'placeId2']
    df_pairs.drop_duplicates(inplace=True)
    df_pairs["detected"] = "yes"

    gold = []
    for i, row in golden_df.iterrows():
        gold.append(sorted([row["placeId1"], row["placeId2"]]))
    gold = pd.DataFrame(gold)
    gold.columns = ['placeId1', 'placeId2']
    gold.drop_duplicates(inplace=True)

    merged_df = pd.merge(gold,
                         df_pairs,
                         how='left',
                         on=['placeId1', 'placeId2'],
                         suffixes=["", "r"])
    not_detected_df = merged_df[merged_df["detected"].isnull()][[
        "placeId1", "placeId2"
    ]]
    not_detected_df.drop_duplicates(inplace=True)

    not_detected_df = pd.merge(not_detected_df,
                               df_raw,
                               how='left',
                               left_on=['placeId1'],
                               right_on=['placeId'])
    not_detected_df.drop('placeId', inplace=True, axis=1)
    not_detected_df = pd.merge(not_detected_df,
                               df_raw,
                               how='left',
                               left_on=['placeId2'],
                               right_on=['placeId'],
                               suffixes=["1", "2"])

    cols = [
        'placeId1', 'placeId2', 'sourceNames1', 'sourceNames2', 'latitude1',
        'latitude2', 'longitude1', 'longitude2',
        'cities1', 'cities2', 'streets1', 'streets2', 'brands1', 'brands2',
        'email1', 'email2', 'houseNumber1', 'houseNumber2',
        'insertedCategories1', 'insertedCategories2', 'internet1', 'internet2',
        'phoneNumbers1', 'phoneNumbers2', 'postalCode1', 'postalCode2',
        'preemptiveCategories1', 'preemptiveCategories2', 'rawCategories1',
        'rawCategories2', 'clusterId1',
        'clusterId2'
    ]
    not_detected_df[cols].to_csv("/workspace/Error_analysis/" + f"Missed_entitymatching_{date_log}_{config.country}.csv",
                                 index=None)

    gold["original"] = "yes"
    merged1 = pd.merge(df_pairs,
                       gold,
                       how='left',
                       on=['placeId1', 'placeId2'],
                       suffixes=["", "r"])
    additional_detected_df = merged1[merged1["original"].isnull()][[
        "placeId1", "placeId2"
    ]]
    additional_detected_df.drop_duplicates(inplace=True)
    additional_detected_df = pd.merge(additional_detected_df,
                                      df_raw,
                                      how='left',
                                      left_on=['placeId1'],
                                      right_on=['placeId'])
    additional_detected_df.drop('placeId', inplace=True, axis=1)
    additional_detected_df = pd.merge(additional_detected_df,
                                      df_raw,
                                      how='left',
                                      left_on=['placeId2'],
                                      right_on=['placeId'],
                                      suffixes=["1", "2"])
    additional_detected_df[cols].to_csv("/workspace/Error_analysis/"
        f"addition_entitymatched_{date_log}_{config.country}.csv", index=None)
