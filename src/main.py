""" Main module for the Robert Parker wine review project """

import argparse
import reviews
import scraper
from loguru import logger
from pathlib import Path
from settings import settings, logger


def get_reviews(country: str) -> None:
    """Get reviews from given country and save to a CSV file"""
    scraper.export_reviews(country, settings)


def main() -> None:
    """Main def to get insights in the reviews"""
    reviews.get_insights(settings)


if __name__ == "__main__":

    # Get the optional country argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country",
        type=str,
        help="The country to export data from Robert Parker.",
        required=False,
    )
    args = parser.parse_args()
    country = args.country

    if country:
        get_reviews(country)
    else:  
        # No data to export. Get insights in already scraped data
        main()
