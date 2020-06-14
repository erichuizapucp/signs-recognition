import os
import logging

from argparse import ArgumentParser
from logger_config import setup_logging
from app.video.video_evaluator import VideoEvaluator


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--video_path', help='Video Path', required=True)

    return parser.parse_args()


def main():
    working_folder = os.getenv('WORK_DIR', './')

    # configure logging
    setup_logging(working_folder, 'app-logging.yaml')
    logger = logging.getLogger(__name__)

    args = get_cmd_args()
    video_path = args.video_path

    logger.debug('the app started with the following parameters: %s', args)

    video_evaluator = VideoEvaluator()
    video_evaluator.evaluate(video_path)

    logger.debug('the app operation is completed')


if __name__ == '__main__':
    main()
