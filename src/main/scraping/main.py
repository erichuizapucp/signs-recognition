import logging
import os

from logger_config import setup_logging
from argparse import ArgumentParser
from scraping.signs_language_scraper import SignsLanguageScraper


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-b', '--base_url', help='Base URL', required=True)
    parser.add_argument('-u', '--url', help='URL', required=True)
    parser.add_argument('-s', '--storage_location', help='Model Name', required=True)

    return parser.parse_args()


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'scraping-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()
    base_url = args.base_url
    url = args.url
    storage_location = args.storage_location

    if base_url:
        logger.debug('Files will be scraped from: %s', base_url)

    if storage_location:
        logger.debug('Scraped files to be stored at: %s', storage_location)

    scraper = SignsLanguageScraper(base_url, storage_location)
    scraper.scrap(url)


if __name__ == '__main__':
    main()
