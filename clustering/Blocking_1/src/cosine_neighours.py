from tqdm.auto import tqdm
import faiss
import numpy as np


def cos_candidates(df, embeddings, batch=1000, K=50, embed_dim=384):
    """Faiss based topk cosine similarity search.

    Args:
        df (dataframe): exploded dataframe
        embeddings (array): array of embeddings
        batch (int, optional): batch size for cosine similarity query. Defaults to 1000.
        K (int, optional): Top k neighbours to search. Defaults to 50.

    Returns:
        similarities,indexes: similarity score and indexes.
    """
    ids = df.Id.values.astype(np.int64)
    index = faiss.IndexFlatIP(embed_dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    topK_indices_each_row = []
    similarities = []
    for i in tqdm(range(0, len(embeddings), batch)):
        emb, indices_each_row = gpu_index.search(embeddings[i : i + batch], k=K)
        topK_indices_each_row.append(indices_each_row)
        similarities.append(emb)
    topK_indices_each_row = np.vstack(topK_indices_each_row)
    similarities = np.vstack(similarities)

    return similarities,topK_indices_each_row
