from Entity_matching.src import (
    add_exact_duplicate,
    inference_all,
    distance_features_em,
    clustering,
    gen_name_embeddings,
    embedding_features,
    combine_features_em
)
import time
import warnings
from Config import config

from utils.logger_helper import log_helper

warnings.filterwarnings("ignore")
from utils.trace import Trace


def entity_match(log):

    log.info(
        "************************** Entity Matching *************************************"
    )
    start_time = time.time()
    trace1 = Trace()

    with trace1.timer("generate_distance_features_v2", log):
        distance_features_em.batch_generate_dist_features(log)
        
    with trace1.timer("generate_embeddings", log):
        gen_name_embeddings.gen_name_embeds(log)
        
    with trace1.timer("generate_embedding_features_v2", log):
        embedding_features.parallel_embedding_features(log)
        
    with trace1.timer("Combining_all_features_entity_matching", log):
        combine_features_em.combine_feat(log)

    with trace1.timer('Forest_inference_em', log):
        inference_all.inference_all_models(log)

    with trace1.timer('Final Clustering', log):
        clustering.cluster(log)

    with trace1.timer('add_exact_duplicates', log):
        add_exact_duplicate.add_exact_dups(log)

    log.info(
        f"Total time taken by Entity Matching = {round((time.time() - start_time)/60,2)} minutes"
    )
    print(
        f"Total time taken by Entity Matching = {round((time.time() - start_time)/60,2)} minutes"
    )


if __name__ == "__main__":
    trace = Trace()
    log = log_helper(f"Entity_matching_{config.country}", config.country)
    with trace.timer("Entity_matching", log):
        entity_match(log)
