import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))
import glob
from pathlib import Path
from Config import config
from tqdm.auto import tqdm
from utils.create_features_em import parallelize_create_edit_features_em,create_edit_features_file_em
from utils.logger_helper import log_helper
from utils.trace import Trace
import warnings
import numpy as np
warnings.filterwarnings("ignore")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

def batch_generate_dist_features(log):
    Path(
        config.root_dir +
        f"Entity_matching/generated_features_em/{config.country}_batch_dist_features_em"
    ).mkdir(parents=True, exist_ok=True)

    file_list = glob.glob(
        config.output_stage1 +
        f"batch_candidates/{config.country}_parquet/*.parquet")

    log.info("Generating edit distance features for entity matching")

    for i in tqdm(range(0, len(file_list), 3)):
        if (len(file_list[i : i + 3])==1):
            df_features = create_edit_features_file_em(file_list[i : i + 3][0])
        else:
            df_features = parallelize_create_edit_features_em(file_list[i : i + 3])
        
        

        df_features.to_parquet(
            config.root_dir
            + f"Entity_matching/generated_features_em/{config.country}_batch_dist_features_em/parquet_distance_{i}.parquet",
            engine="pyarrow",
            compression="zstd",
            index=None,
        )
    log.info("completed generating distance features for entity matching")
    print("completed generating distance features for entity matching")


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(
        f"generate_distance_features_entity_matching_{config.country}",
        config.country)
    with trace.timer("generate_distance_features_v2", log):
        batch_generate_dist_features(log)
