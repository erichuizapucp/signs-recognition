import os
import shutil

from argparse import ArgumentParser
from learning.dataset.prepare.swav.swav_video_dataset_preparer import SwAVDatasetPreparer
from learning.dataset.preview.swav_dataset_previewer import SwAVDatasetPreviewer
from learning.common.dataset_type import SWAV

from learning.common.object_detection_utility import ObjectDetectionUtility


def get_cmd_args():
    parser = ArgumentParser()

    parser.add_argument('-dt', '--dataset_type', help='Dataset Type', required=True)
    parser.add_argument('-tr', '--train_dataset_path', help='Train Dataset Path', required=True)
    parser.add_argument('-l', '--logs_folder', help='Logs Folder', required=True)
    parser.add_argument('-odm', '--object_detection_model_name', help='Object Detection Model Name', required=False)
    parser.add_argument('-odcp', '--object_detection_checkout_prefix', help='Object Detection Checkout Prefix',
                        required=False)

    return parser.parse_args()


def get_dataset_preparer(dataset_type, dataset_path, **kwargs):
    preparers = {
        SWAV: lambda: SwAVDatasetPreparer(train_dataset_path=dataset_path,
                                          test_dataset_path=None,
                                          object_detection_model=kwargs['ObjectDetectionModel'])
    }
    preparer = preparers[dataset_type]()
    return preparer


def get_dataset_previewer(dataset_type, log_folder, preparer):
    previewers = {
        SWAV: lambda: SwAVDatasetPreviewer(log_folder, preparer)
    }
    previewer = previewers[dataset_type]()
    return previewer


def main():
    args = get_cmd_args()

    dataset_type = args.dataset_type
    train_dataset_path = args.train_dataset_path
    logs_folder = args.logs_folder
    object_detection_model_name = args.object_detection_model_name
    object_detection_checkout_prefix = args.object_detection_checkout_prefix

    if os.path.exists(logs_folder):
        shutil.rmtree(logs_folder)

    object_detection_utility = ObjectDetectionUtility()
    object_detection_model = object_detection_utility.get_object_detection_model(object_detection_model_name,
                                                                                 object_detection_checkout_prefix)

    # Object Detection is needed to focus only on person images and not on other elements in video frames that
    # are not relevant for our use case.
    preparer = get_dataset_preparer(dataset_type, train_dataset_path, ObjectDetectionModel=object_detection_model)
    previewer = get_dataset_previewer(dataset_type, logs_folder, preparer)
    previewer.preview_dataset()


if __name__ == '__main__':
    main()
