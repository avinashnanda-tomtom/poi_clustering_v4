import time
import warnings

from clustering.Blocking_1.blocking_stage1 import block1
from clustering.Entity_matching.entitymatching import entity_match
from Config import config
from utils.logger_helper import log_helper

warnings.filterwarnings("ignore")
from utils.trace import Trace

if __name__ == "__main__":
    start_time = time.time()
    trace = Trace()
    log = log_helper(f"Poi_clustering_{config.country}", config.country)

    with trace.timer("Blocking_stage_1", log):
        block1(log)

    with trace.timer("Entity_matching", log):
        entity_match(log)

    print(
        f"Total time taken by Poi Clustering = {round((time.time() - start_time)/60,2)} minutes"
    )

    log.info(
        f"Total time taken by Poi Clustering = {round((time.time() - start_time)/60,2)} minutes"
    )
