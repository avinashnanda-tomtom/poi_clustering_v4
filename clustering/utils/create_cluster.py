import numpy as np
import pandas as pd


def match_clusterIds(df2, threshold=0.5, group_link_threshold=0.6):
    """Create clusters based on threshold.
    Args:
        df2 (dataframe): dataframe of predictions
        threshold (float, optional): threshold to link two ids. Defaults to 0.5.
        group_link_threshold (float, optional): Threshold to add a id to already existing cluster. Defaults to 0.6.

    Returns:
        _type_: id clusterid dict
    """
    matches = df2[df2["Prediction"] > threshold].reset_index(drop=True)

    id1, id2, preds = np.split(matches.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()

    id_to_clusterId = {}  # maps each ID to POI
    clusterId_counts = (
        {}
    )  # counts number of IDs in each POI - used for threshold determination
    clusterId_to_id = {}

    clusterId = 0
    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if (i1 in id_to_clusterId) and (i2 in id_to_clusterId):
            # Merging will be handled later
            continue

        # i1 is already in dict - assign i2 to the same clusterId
        elif i1 in id_to_clusterId:
            if pred > group_link_threshold:
                id_to_clusterId[i2] = id_to_clusterId[i1]
                clusterId_to_id[id_to_clusterId[i1]].append(i2)
                clusterId_counts[id_to_clusterId[i1]] += 1

        # i2 is already in dict - assign i1 to the same clusterId
        elif i2 in id_to_clusterId:
            if pred > group_link_threshold:
                id_to_clusterId[i1] = id_to_clusterId[i2]
                clusterId_to_id[id_to_clusterId[i2]].append(i1)
                clusterId_counts[id_to_clusterId[i2]] += 1

        # New POI
        else:
            id_to_clusterId[i1] = clusterId
            id_to_clusterId[i2] = clusterId

            clusterId_to_id[clusterId] = [i1, i2]

            clusterId_counts[clusterId] = 2
            clusterId += 1

    return id_to_clusterId, clusterId_to_id, clusterId_counts


def merge_clusters_simple(
    df2,
    id_to_clusterId,
    clusterId_to_id,
    clusterId_counts,
    threshold=0.5,
    threshold_merge=0.9,
    max_size=300,
):
    """merge clusters if a common id prediction is > threshold_merge.

    Args:
        df2 (dataframe): _description_
        id_to_clusterId (dict): _description_
        clusterId_to_id (dict): _description_
        clusterId_counts (dict): _description_
        threshold (float, optional): single link threshold. Defaults to 0.5.
        threshold_merge (float, optional): threshold to merge on. Defaults to 0.9.
        max_size (int, optional): maximum cluster size. Defaults to 300.

    Returns:
        _type_: _description_
    """

    matches = df2[df2["Prediction"] > threshold].reset_index(drop=True)

    id1, id2, preds = np.split(matches.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()

    for i, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if i1 in id_to_clusterId and i2 in id_to_clusterId:
            # only merge if combined size is <= 300 and pred > th3
            if (
                id_to_clusterId[i2] != id_to_clusterId[i1]
                and pred > threshold_merge
                and clusterId_counts[id_to_clusterId[i1]]
                + clusterId_counts[id_to_clusterId[i2]]
                <= max_size
            ):
                m = min(id_to_clusterId[i2], id_to_clusterId[i1])
                m2 = max(id_to_clusterId[i2], id_to_clusterId[i1])

                clusterId_counts[m] = (
                    clusterId_counts[id_to_clusterId[i1]]
                    + clusterId_counts[id_to_clusterId[i2]]
                )
                clusterId_counts[m2] = 0

                for j in clusterId_to_id[m2]:
                    id_to_clusterId[j] = m

                clusterId_to_id[m] = clusterId_to_id[m] + clusterId_to_id[m2]
                clusterId_to_id[m2] = []

    return id_to_clusterId, clusterId_to_id, clusterId_counts


def merge_clusters_advanced(
    df2,
    id_to_clusterId,
    clusterId_to_id,
    clusterId_counts,
    threshold_merge_avg=0.9,
    threshold_merge_max=0.95,
    max_size=300,
):
    """merge cluster advance based on below condition:
    if prediction > threshold_merge_max and count of common ids >50% in each group then merge.
    if mean prediction of group > threshold_merge_avg and group size >1 then merge.

    Args:
        df2 (dataframe): pairs dataframe with prediction
        id_to_clusterId (dict): map between id and clusterid
        clusterId_to_id (dict): map between clusterid and id
        clusterId_counts (dict): map of count of ids in a clusterid
        threshold_merge_avg (float, optional): mean predicition of group to merge. Defaults to 0.9.
        threshold_merge_max (float, optional): _description_. Defaults to 0.95.
        max_size (int, optional): maximum cluster size. Defaults to 300.

    Returns:
        _type_: _description_
    """
    id1, id2, preds = np.split(df2.values, [1, 2], axis=1)
    id1, id2, preds = id1.flatten(), id2.flatten(), preds.flatten()
    merging_pairs = []

    for _, (i1, i2, pred) in enumerate(zip(id1, id2, preds)):
        if i1 in id_to_clusterId and i2 in id_to_clusterId:
            clusterId1 = id_to_clusterId[i1]
            clusterId2 = id_to_clusterId[i2]

            if id_to_clusterId[i2] == id_to_clusterId[i1]:
                continue

            if clusterId_counts[clusterId1] + clusterId_counts[clusterId2] > max_size:
                continue  # Too big, skip

            m = min(clusterId1, clusterId2)
            m2 = max(clusterId2, clusterId1)
            to_merge = [m, m2, i1, i2, pred]
            merging_pairs.append(to_merge)

    df_merge = pd.DataFrame(
        merging_pairs, columns=["clusterId1", "clusterId2", "i1", "i2", "score"]
    )
    df_merge["clusterId1_clusterId2"] = (
        df_merge["clusterId1"].astype(str) + "_" + df_merge["clusterId2"].astype(str)
    )

    to_merge = {}

    for _, merge in df_merge.groupby("clusterId1_clusterId2"):

        m, m2, i1, i2 = merge[["clusterId1", "clusterId2", "i1", "i2"]].values[0]

        s1 = clusterId_counts[id_to_clusterId[i1]]
        s2 = clusterId_counts[id_to_clusterId[i2]]
        links_prop = len(merge) / min(s1, s2)

        if (
            (merge["score"].max() > threshold_merge_max)
            and (links_prop > 0.50)
            or (merge["score"].mean() > threshold_merge_avg and len(merge) > 1)
        ):
            try:
                to_merge[m2] = to_merge[m]
            except KeyError:
                to_merge[m2] = m

    for m2, m in to_merge.items():
        if clusterId_counts[m] + clusterId_counts[m2] > max_size:
            continue

        clusterId_counts[m] = clusterId_counts[m] + clusterId_counts[m2]
        clusterId_counts[m2] = 0

        for clusterId in clusterId_to_id[m2]:
            id_to_clusterId[clusterId] = m

        clusterId_to_id[m] = clusterId_to_id[m] + clusterId_to_id[m2]
        clusterId_to_id[m2] = []

    return id_to_clusterId, clusterId_to_id, clusterId_counts


def agglomerative_clustering(
    df_predicition,
    prediction,
    single_link_threshold=0.90,
    group_link_threshold=0.90,
    threshold_merge_avg=0.90,
    threshold_merge=0.95,
    max_size=400,
):
    """Agglomerative clustering based on threshold.

    Args:
        df_predicition (dataframe): pair dataframe
        prediction (list): prediction probability of the pairs
        single_link_threshold (float, optional): probability value above which put to poi in a cluster. Defaults to 0.90.
        group_link_threshold (float, optional): probability value above which add a poi to a already created cluster. Defaults to 0.90.
        threshold_merge_avg (float, optional): mean probability value above which merge two clusters. Defaults to 0.9.
        threshold_merge (float, optional): maximum probaility value above which merge two clusters.. Defaults to 0.98.
        max_size (int, optional): maximum size of clusters. Defaults to 400.

    Returns:
        _type_: dataframe of id and clusterids
    """

    df2 = df_predicition.copy()
    df2["Prediction"] = np.copy(prediction)

    try:
        df2 = df2[["id_1", "id_2", "Prediction"]]
        df2.columns = ["id", "id2", "Prediction"]
    except KeyError:
        df2 = df2[["id", "id2", "Prediction"]]

    # sort by decr prediction
    df2 = df2.sort_values(by=["Prediction"], ascending=False).reset_index(drop=True)

    # Build clusters
    id_to_clusterId, clusterId_to_id, clusterId_counts = match_clusterIds(
        df2, threshold=single_link_threshold, group_link_threshold=group_link_threshold
    )

    # Merge clusters

    if threshold_merge_avg > 0:
        id_to_clusterId, clusterId_to_id, clusterId_counts = merge_clusters_advanced(
            df2,
            id_to_clusterId,
            clusterId_to_id,
            clusterId_counts,
            threshold_merge_avg=threshold_merge_avg,
            threshold_merge_max=threshold_merge,
            max_size=max_size,
        )
    else:
        id_to_clusterId, clusterId_to_id, clusterId_counts = merge_clusters_simple(
            df2,
            id_to_clusterId,
            clusterId_to_id,
            clusterId_counts,
            threshold=single_link_threshold,
            threshold_merge=threshold_merge,
            max_size=max_size,
        )

    # Reformat
    preds = pd.DataFrame.from_dict(id_to_clusterId, orient="index").reset_index()
    preds.columns = ["id", "clusterId"]

    return preds
