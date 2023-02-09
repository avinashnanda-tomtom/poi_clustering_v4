from utils.trace import Trace
from utils.logger_helper import log_helper
from sentence_transformers import SentenceTransformer
from Config import config
import pandas as pd
import joblib
import os
from pathlib import Path
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gen_name_embeds(log):
    Path(config.embed_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
        dtype={"postalCode": "str", "houseNumber": "str"},
    )
    for c in ["sourceNames"]:
        df[c] = df[c].map(str)



    log.info("generating sourcename embeddings")
    
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 64

    # with torch.cuda.amp.autocast(enabled=True):
    embeddings = model.encode(list(df["sourceNames"]), batch_size=32, show_progress_bar=True,normalize_embeddings=True)

    with open(config.embed_dir + f"embeddings_name_{config.country}.pkl", "wb") as fo:
        joblib.dump(embeddings, fo)

    log.info("completed generating all embeddings")

    print("completed generating all embeddings")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Generating_name_embeddings_{config.country}", config.country)

    with trace.timer("Generating_name_embeddings", log):
        gen_name_embeds(log)
