import logging
from collections import OrderedDict


logger_initialized = OrderedDict()


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = f"%(name)s - (%(filename)s:%(lineno)d) - %(levelname)s - %(asctime)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def get_logger(name, with_stream=True, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the logger by adding one or two handlers
    otherwise the initialized logger will be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler will be added to the logger.
        log_level (int): The logger level. Note that only the process of rank 0 is affected, and other processes will
            set the level to "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    if len(logger_initialized) == 0:
        logging.basicConfig(level=logging.ERROR, handlers=[])

    logger = logging.getLogger(name)
    # e.g., logger "a" is initialized, then logger "a.b" will skip the initialization since it is a child of "a".
    for logger_name, logger_level in logger_initialized.items():
        if name.startswith(logger_name):
            logger.setLevel(logger_level)
            return logger

    logger.propagate = False
    handlers = []

    if with_stream:
        handlers.append(logging.StreamHandler())

    formatter = CustomFormatter(datefmt="%Y-%m-%d %H:%M")
    log_fmt = f"%(name)s - (%(filename)s:%(lineno)d) - %(levelname)s - %(asctime)s - %(message)s"
    file_formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d,%H:%M:%S")

    logger.handlers = []

    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(file_formatter)
            logger.addHandler(handler)
        else:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = log_level
    return logger


logger = get_logger("real_robot")
