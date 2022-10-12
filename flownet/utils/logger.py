import logging
import os
import sys


def create_logger(logger_name, level=logging.INFO, add_console_handler=True, file_handler_path="", use_formatter=False):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    format_string = "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s: %(lineno)d — %(message)s"
    log_format = logging.Formatter(format_string)

    # Creating and adding the console handler
    if add_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        if use_formatter:
            console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(os.path.join(file_handler_path, f"{logger_name}.log"), mode='w')
    if use_formatter:
        file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
