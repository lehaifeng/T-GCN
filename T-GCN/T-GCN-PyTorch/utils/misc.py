import logging


def format_logger(logger, format='\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s'):
    handler = logger.handlers[0]
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
