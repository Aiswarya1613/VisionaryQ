import logging
import sys


LOGGER_NAME = "visionaryq"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO
) -> None:
    """
    Configure VisionaryQ application logging.

    Configuration is applied once per Python process.
    """

    root_logger = logging.getLogger(
        LOGGER_NAME
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler(
            sys.stdout
        )

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT
        )

        handler.setFormatter(
            formatter
        )

        root_logger.addHandler(
            handler
        )

    root_logger.setLevel(
        level
    )

    root_logger.propagate = False


def get_logger(
    component: str
) -> logging.Logger:
    """
    Return a named VisionaryQ logger.
    """

    configure_logging()

    return logging.getLogger(
        f"{LOGGER_NAME}.{component}"
    )