from utils.trace import Trace
from clustering.Blocking_1.src import (
    explode_clean,
    creating_gt,
    cosine_blocking,
    gen_embeddings,
    combine_blocks,
    housenumber_block,
    phonenumber_block,
    sourcename_block,
    remove_exact_duplicates
)
import time
import warnings
from Config import config
from utils.logger_helper import log_helper

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def block1(log):
    log.info(
        "************************** Blocking stage 1 *************************************"
    )

    start_time = time.time()
    trace1 = Trace()

    # with trace1.timer("drop_exact_duplicates", log):
    #     remove_exact_duplicates.remove_exact_dups(log)

    # with trace1.timer("explode_and_clean", log):
    #     explode_clean.explode_clean(log)

    # with trace1.timer("Generating_ground_truth", log):
    #     creating_gt.create_gtruth(log)

    # with trace1.timer("Generating_sbert_embeddings", log):
    #     gen_embeddings.gen_embeddings(log)

    # with trace1.timer("generate_cosine_candidates", log):
    #     cosine_blocking.gen_cos_candidates(log)

    # with trace1.timer("generate_house_candidates", log):
    #     housenumber_block.house_block(log)

    # with trace1.timer("generate_phone_candidates", log):
    #     phonenumber_block.phone_block(log)

    # with trace1.timer("generate_name_candidates", log):
    #     sourcename_block.name_block(log)

    with trace1.timer("combine_blocks", log):
        combine_blocks.combine(log)

    log.info(
        f"Total time taken by blocking1 = {round((time.time() - start_time)/60,2)} minutes"
    )
    print(
        f"Total time taken by blocking1 = {round((time.time() - start_time)/60,2)} minutes"
    )
1

if __name__ == "__main__":
    start_time = time.time()
    trace = Trace()
    log = log_helper(f"Blocking_stage_1_{config.country}", config.country)
    with trace.timer("Blocking_stage_1", log):
        block1(log)
