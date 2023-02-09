import logging
import pathlib
from datetime import datetime


def log_helper(fileName, country, logLevel=logging.INFO):
    """Function to create logs .

    Args:
        fileName (text): Filename of log file
        country (text): country/folder to create log file in.
        logLevel (_type_, optional): logging level . Defaults to logging.INFO.

    Returns:
        logger: logger
    """
    date_log = datetime.now().strftime("%d_%m_%Y")
    pathlib.Path(f"/workspace/clustering/logs/{country}/{country}_{date_log}").mkdir(
        parents=True, exist_ok=True
    )
    logging.basicConfig(level=logLevel)
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)
    filename = f"/workspace/clustering/logs/{country}/{country}_{date_log}/{fileName}_{date_log}.log"
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
