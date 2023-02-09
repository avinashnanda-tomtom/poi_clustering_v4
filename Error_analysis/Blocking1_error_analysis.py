import pandas as pd
from Config import config
import glob
from tqdm.auto import tqdm
import numpy as np
from utils.trace import reduce_mem_usage

if __name__ == "__main__":
    golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")
    df_raw = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv",engine='c')
    candidate_set_df = pd.DataFrame()

    for file in tqdm(glob.glob(config.output_stage1 + f"batch_candidates/{config.country}_parquet/*.parquet")):
        df = pd.read_parquet(file,engine="pyarrow")
        candidate_set_df = pd.concat([candidate_set_df,df],axis=0, ignore_index=True)
        
    candidate_set_df = reduce_mem_usage(candidate_set_df)
       
       
    cols_list = ["placeId1","placeId2"]
    tmp_arr = np.array(candidate_set_df.loc[:, cols_list])
    tmp_arr.sort(axis=1)
    cand  = pd.DataFrame(tmp_arr)
    cand.columns = cols_list
    cand.drop_duplicates(inplace=True)
    cand["present"] = True
    
    tmp_arr = np.array(golden_df.loc[:, cols_list])
    tmp_arr.sort(axis=1)
    gold  = pd.DataFrame(tmp_arr)
    gold.columns = cols_list
    gold.drop_duplicates(inplace=True)
    

            
    merged_df = pd.merge(gold,
                            cand,
                            how='left',
                            on=['placeId1', 'placeId2'],
                            suffixes=["", "r"])
    not_detected_df = merged_df[merged_df["present"].isnull()]
    not_detected_df = not_detected_df[["placeId1","placeId2"]]
    not_detected_df.drop_duplicates(inplace=True)
    
    print(f"total duplicate not detected {len(not_detected_df)}")
    
    not_detected_df = pd.merge(not_detected_df, df_raw,  how='left', left_on=[
                    'placeId1'], right_on=['placeId'])
    not_detected_df.drop('placeId',inplace=True,axis=1)
    not_detected_df = pd.merge(not_detected_df, df_raw,  how='left', left_on=[
                        'placeId2'], right_on=['placeId'],suffixes=["1","2"])
    cols = [
        'placeId1', 'placeId2', 'sourceNames1', 'sourceNames2', 'latitude1',
        'latitude2', 'longitude1', 'longitude2',
        'cities1', 'cities2', 'streets1', 'streets2', 'brands1', 'brands2',
        'email1', 'email2', 'houseNumber1', 'houseNumber2',
        'insertedCategories1', 'insertedCategories2', 'internet1', 'internet2',
        'phoneNumbers1', 'phoneNumbers2', 'postalCode1', 'postalCode2',
        'preemptiveCategories1', 'preemptiveCategories2', 'rawCategories1',
        'rawCategories2','clusterId1',
        'clusterId2'
    ]
    
    not_detected_df[cols].to_csv("/workspace/Error_analysis/" + f"Missed_blocking1_{config.country}.csv",index=None)