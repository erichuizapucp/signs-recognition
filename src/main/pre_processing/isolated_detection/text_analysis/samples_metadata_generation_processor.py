import os
import csv
import json
import logging


from pre_processing.common import io_utils
from pre_processing.common.processor import Processor


class SamplesMetadataGenerationProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.metadata_file_name = 'samples-generation-metadata.csv'
        self.excluded_folder = "excluded/"
        self.override_local_files = False
        self.min_instances_per_sample = 10

        self.transcription_path_prefix = 'transcriptions'
        self.video_extension = '.mp4'

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        syntax_local_folder_path = os.path.join(self.work_dir, data['syntax']['localFolderPath'])
        syntax_s3_folder_path = data['syntax']['s3Path']

        transcription_local_folder_path = os.path.join(self.work_dir, data['transcription']['localFolderPath'])
        transcription_s3_folder_path = data['transcription']['s3Path']

        video_path_prefix = data['output']['video_path_prefix']

        if self.override_local_files:
            # download syntax detection files (e.g. nouns and numbers)
            self.s3_batch_download_files(syntax_s3_folder_path, self.excluded_folder, allowed_extensions=['.csv'])
            # download video_transcription files
            self.s3_batch_download_files(transcription_s3_folder_path, self.excluded_folder, allowed_extensions=['.json'])

        # load syntax tokens (nouns and numbers) in a list
        syntax_tokens = self.get_syntax_tokens(syntax_local_folder_path)

        # create the samples metadata file
        metadata_file_columns = ['token', 'video_name', 'start_time', 'end_time']
        metadata_file_path = self.create_metadata_file(metadata_file_columns, syntax_local_folder_path)
        self.logger.debug('Samples Metadata File created at: %s', metadata_file_path)

        # find syntax occurrence within video_transcription files
        transcription_paths = io_utils.get_files_in_folder(self.work_dir,
                                                           transcription_local_folder_path, "*.json")
        for transcription_file_path in transcription_paths:
            self.logger.debug('Processing %s', transcription_file_path)

            transcription_items = self.get_transcription_items(transcription_file_path)

            transcription_file_relative_path = transcription_file_path.replace(self.work_dir + '/', '')
            self.handle_transcription_items(transcription_items, syntax_tokens, transcription_file_relative_path,
                                            video_path_prefix, metadata_file_path, metadata_file_columns)

        self.upload_file(metadata_file_path, syntax_s3_folder_path + self.metadata_file_name)
        self.logger.debug('samples generation metadata uploaded to: %s', syntax_s3_folder_path + self.metadata_file_name)

    def get_syntax_tokens(self, local_folder_path):
        syntax_tokens = []

        syntax_paths = io_utils.get_files_in_folder(self.work_dir, local_folder_path, "*.csv")
        for syntax_file_local_path in syntax_paths:
            if self.metadata_file_name in syntax_file_local_path:
                continue

            with open(syntax_file_local_path, 'r') as syntax_file:
                reader = csv.reader(syntax_file)
                next(reader, None)
                for syntax_entry in reader:
                    # only match samples with at least 10 instances
                    if int(syntax_entry[1]) >= self.min_instances_per_sample:
                        syntax_tokens.append(syntax_entry[0])

        return syntax_tokens

    @staticmethod
    def get_transcription_items(transcription_file_path):
        with open(transcription_file_path, 'r') as transcription_file:
            transcription_data = json.load(transcription_file)
            transcription_items = [transcription_item for transcription_item in transcription_data['results']['items']
                                   if transcription_item['type'] == 'pronunciation' and
                                   'speaker_label' in transcription_item and
                                   transcription_item['speaker_label'] == 'spk_1']

        return transcription_items

    def create_metadata_file(self, file_columns, syntax_local_folder_path):
        metadata_file_path = os.path.join(syntax_local_folder_path, self.metadata_file_name)
        with open(metadata_file_path, 'w') as metadata_file:
            wr = csv.DictWriter(metadata_file, fieldnames=file_columns)
            wr.writeheader()

        return os.path.join(self.work_dir, metadata_file_path)

    def handle_transcription_items(self, transcription_items, syntax_tokens, transcription_file_path,
                                   video_path_prefix, metadata_file_path, file_columns):

        with open(metadata_file_path, 'a') as metadata_file:
            wr = csv.DictWriter(metadata_file, fieldnames=file_columns)

            for transcription_item in transcription_items:
                content = transcription_item['alternatives'][0]['content']
                if content in syntax_tokens:
                    self.logger.debug('Found a match: %s', content)
                    video_local_path = io_utils.change_extension(transcription_file_path, self.video_extension)
                    video_path = os.path.basename(video_local_path)

                    metadata_object = {
                        'token': content,
                        'video_name': video_path,
                        'start_time': transcription_item['start_time'],
                        'end_time': transcription_item['end_time']
                    }

                    wr.writerow(metadata_object)
