import logging
import os

def setup_log(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, 'log.log')

    logger = logging.getLogger()

    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(file_name)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)


def log_config(config, logger):
    """
    Logs the configuration attributes line by line using the provided logger.

    :param config: A configuration object containing attributes to be logged.
    :param logger: A logger object to use for logging.
    """
    config_dict = config.__dict__
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
