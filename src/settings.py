""" Settings module for the Robert Parker project """

import sys
from loguru import logger
from pathlib import Path
from pydantic import BaseSettings


class _Settings(BaseSettings):
    """Settings class for the project. To be instantiated once
    only.
    """

    base_dir: Path = Path(__file__).parents[1]
    data_dir: Path = base_dir / "data"
    images_dir: Path = base_dir / "images"
    log_dir: Path = base_dir / "log"

    url: str = rf"https://www.robertparker.com"

    raw_data_dir: Path = data_dir / "raw"
    cleaned_data_file: Path = data_dir / "cleaned" / "cleaned_wines.csv"

    rating_col: str = "rating"


settings = _Settings()

logger.add(
    sink=settings.log_dir / "log_{time:YYYYMMDD}.log",
    rotation="00:00",
    enqueue=True
)
# logger.add(
#     sink=sys.stdout,                    # The screen (standard output 'channel').
#     colorize=True,                      # Add colors to the messages.
#     format="<green>{time}</green> <level>{message}</level>"
# )
# logger.add(
#     sink=sys.stderr,                    # The screen (standard error 'channel').
#     colorize=True,                      
#     format="<red>{time}</red> <level>{message}</level>"
# )
