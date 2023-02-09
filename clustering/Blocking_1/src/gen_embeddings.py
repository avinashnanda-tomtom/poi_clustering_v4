import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))
import joblib
import pandas as pd
from Config import config
from sentence_transformers import SentenceTransformer
from utils.logger_helper import log_helper
from utils.trace import Trace
from pathlib import Path
import torch
from utils.cleaning_utils import clean_text

def gen_embeddings(log):
    Path(config.embed_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        config.input_dir + f"Fuse_exploded_{config.country}_cleaned.csv",engine='c',
        dtype={"postalCode": "str", "houseNumber": "str"},
    )
    
    df["phoneNumbers"] = df["phoneNumbers"].apply(clean_text)
    
    cols_to_block = [
        "sourceNames", "latitude", "longitude", "category", "subCategory",
        "phoneNumbers", "houseNumber", "streets", "cities", "brands",
        "postalCode"
    ]


    for col in cols_to_block:
        df[col] = df[col].fillna("")

    df["all"] = df[cols_to_block].astype(str).agg(" ".join, axis=1)

    with torch.cuda.amp.autocast(enabled=True):
        model = SentenceTransformer("all-MiniLM-L12-v2")

        embeddings = model.encode(list(df["all"]),
                                batch_size=512,
                                show_progress_bar=True,
                                normalize_embeddings=True)

    with open(config.embed_dir + f"embeddings_{config.country}.pkl",
              "wb") as fo:
        joblib.dump(embeddings, fo)

    log.info("Completed Generating embeddings")
    print("Completed Generating embeddings")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Generating_embeddings_{config.country}", config.country)

    with trace.timer("Generating_embeddings", log):
        gen_embeddings(log)
