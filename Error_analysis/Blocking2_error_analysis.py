import pandas as pd
from Config import config
import glob
from tqdm.auto import tqdm


if __name__ == "__main__":
    golden_df = pd.read_csv(config.gt_dir + f"GT_{config.country}.csv")
    df_raw = pd.read_csv(config.raw_dir + f"Fuse_{config.country}.csv",engine='c')
    candidate_set_df = pd.DataFrame()

    for file in tqdm(glob.glob(config.output_stage2 + f"batch_candidates/{config.country}_parquet/*.parquet")):
        df = pd.read_parquet(file,engine="pyarrow")
        candidate_set_df = pd.concat([candidate_set_df,df],axis=0, ignore_index=True)
        
    merged_df = pd.merge(golden_df,candidate_set_df,how ='left',
                        on=['ltable_id', 'rtable_id'],suffixes=["","r"])
    
    not_detected_df = merged_df[merged_df["placeId1r"].isnull()]
    not_detected_df = not_detected_df[["placeId1","placeId2"]]
    not_detected_df.drop_duplicates(inplace=True)
    
    print(f"total duplicate not detected {len(not_detected_df)}")
    
    not_detected_df = pd.merge(not_detected_df, df_raw,  how='left', left_on=[
                    'placeId1'], right_on=['placeId'])
    not_detected_df.drop('placeId',inplace=True,axis=1)
    not_detected_df = pd.merge(not_detected_df, df_raw,  how='left', left_on=[
                        'placeId2'], right_on=['placeId'],suffixes=["1","2"])
    
    not_detected_df.to_csv("/workspace/Error_analysis/" + f"Missed_blocking2_{config.country}.csv",index=None)