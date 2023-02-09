import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import psutil



def compute_blocking_statistics_placeid(candidate_set_df, golden_df):

    merged_df = pd.merge(
        candidate_set_df, golden_df, on=["placeId1", "placeId2"], suffixes=["", "r"]
    )
    total_duplicates = golden_df[["placeId1", "placeId2"]].drop_duplicates().shape[0]
    duplicate_detected = merged_df[["placeId1", "placeId2"]].drop_duplicates().shape[0]

    statistics_dict = {
        "candidate_set_length": len(
            candidate_set_df[["placeId1", "placeId2"]].drop_duplicates()
        ),
        "precision": duplicate_detected
        / len(candidate_set_df[["placeId1", "placeId2"]].drop_duplicates()),
        "recall": duplicate_detected / total_duplicates,
    }
    return statistics_dict


def compute_blocking_statistics(candidate_set_df, golden_df, df,join_placeId=False):
    """Compute metrics for blocking.

    Args:
        candidate_set_df (dataframe): Dataframe of pairs from blocking.
        golden_df (dataframe): Dataframe containing true pairs.
        df (dataframe): Exploded dataframe.

    Returns:
        Dictionary: dictionary of metrics.
    """
    
    if join_placeId:
        print("Joining the ids with placeId")
        placeId = df["placeId"].to_numpy()
        candidate_set_df["placeId1"] = placeId[candidate_set_df['ltable_id'].to_numpy()]
        candidate_set_df["placeId2"] = placeId[candidate_set_df['rtable_id'].to_numpy()]
    
    
    cols_list = ["placeId1","placeId2"]
    tmp_arr = np.array(candidate_set_df.loc[:, cols_list])
    tmp_arr.sort(axis=1)
    cand  = pd.DataFrame(tmp_arr)
    cand.columns = cols_list
    cand.drop_duplicates(inplace=True)
    
    tmp_arr = np.array(golden_df.loc[:, cols_list])
    tmp_arr.sort(axis=1)
    gold  = pd.DataFrame(tmp_arr)
    gold.columns = cols_list
    gold.drop_duplicates(inplace=True)
    
    merged_df = pd.merge(
        cand, gold, on=["placeId1", "placeId2"], suffixes=["", "r"]
    )
    total_duplicates = gold[["placeId1", "placeId2"]].drop_duplicates().shape[0]
    duplicate_detected = merged_df[["placeId1", "placeId2"]].drop_duplicates().shape[0]

    statistics_dict = {
        "candidate_set_length": len(candidate_set_df),
        "pair_entity_ratio": len(candidate_set_df) / len(df),
        "precision": duplicate_detected / len(candidate_set_df),
        "recall": duplicate_detected / total_duplicates,
    }

    return statistics_dict


def clean_placeid(candidate_df, df):
    """Remove the candidates where placeid are same.

    Args:
        candidate_df (dataframe): Dataframe of pairs from blocking.
        df (dataframe): Exploded dataframe.

    Returns:
        dataframe: dataframe without same placeId as candidates.
    """
    print(f"The total number of candidates  = {len(candidate_df)}")
    print("Joining the ids with placeId")
    placeId = df["placeId"].to_numpy()
    candidate_df["placeId1"] = placeId[candidate_df['ltable_id'].to_numpy()]
    candidate_df["placeId2"] = placeId[candidate_df['rtable_id'].to_numpy()]
    print("cleaning where placeId are same")
    candidate_df = candidate_df[~(candidate_df["placeId1"] == candidate_df["placeId2"])]
    print(
        f"The total number of candidates after removing place ids = {len(candidate_df)}"
    )
    return candidate_df


def chunk_size(size):
    """Calculate how many partitions to create of a dataframe for better multiprocessing

    Args:
        size (_type_): size of dataframe.

    Returns:
        int: number of rows in each partition.
    """

    memory = psutil.virtual_memory().total / 2.0**30

    if memory > 64:
        chunk_thresh = 4000000
    else:
        chunk_thresh = 2000000

    for i in range(8, 100000000000, 4):
        if int(size / 4) < chunk_thresh:
            c_size = int(size / 4)
            return c_size
        if size / i < chunk_thresh:
            c_size = int(size / i)
            return c_size
        else:
            continue




def create_pairs_cosine(df, topK_neighbors,similarities):
    """Create pairs from output of cosine similarity.

    Args:
        df (dataframe): exploded dataframe
        topK_neighbors (indexes): indexes of nearest neighbours.
        similarities (float): cosine similarity value.

    Returns:
        _type_: _description_
    """
    ids = df.Id.values.astype(np.int64)
    all_combination = set()

    for i in tqdm(ids):
        for j in range(0, topK_neighbors.shape[1]):
            if i != topK_neighbors[i][j]:
                all_combination.add(
                    tuple(sorted([i, topK_neighbors[i][j]]) + [similarities[i][j]])
                )
    return all_combination
