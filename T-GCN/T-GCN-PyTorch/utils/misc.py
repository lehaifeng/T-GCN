import logging


def format_logger(logger_name='lightning', format='\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s'):
    logger = logging.getLogger(logger_name)
    handler = logger.handlers[0]
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
