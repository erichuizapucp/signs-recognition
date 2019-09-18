import logging
import json
import csv
import os

from common.processor import Processor
from common import io_utils


class SamplesMetadataGenerationProcessor(Processor):
    __metadata_file_name = 'samples-generation-metadata.csv'

    def __init__(self):
        super().__init__()

        self.__syntax_local_folder_path = None
        self.__syntax_s3_folder_path = None
        self.__transcription_local_folder_path = None
        self.__transcription_s3_folder_path = None

        self.__override_local_files = False

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        self.__syntax_local_folder_path = data['syntax']['localFolderPath']
        self.__syntax_s3_folder_path = data['syntax']['s3Path']

        self.__transcription_local_folder_path = data['transcription']['localFolderPath']
        self.__transcription_s3_folder_path = data['transcription']['s3Path']

        if self.__override_local_files:
            # download syntax detection files (e.g. nouns and numbers)
            self.s3_batch_download_files(self.__syntax_s3_folder_path, allowed_extensions=['.csv'])
            # download transcription files
            self.s3_batch_download_files(self.__transcription_s3_folder_path, allowed_extensions=['.json'])

        # load syntax tokens (nouns and numbers) in a list
        syntax_tokens = self.__get_syntax_tokens(self.__syntax_local_folder_path)

        # create the samples metadata file
        metadata_file_columns = ['token', 'video_path', 'start_time', 'end_time']
        metadata_file_path = self.__create_metadata_file(metadata_file_columns)
        self.logger.debug('Samples Metadata File created at: %s', metadata_file_path)

        # find syntax occurrence within transcription files
        transcription_paths = io_utils.get_files_in_folder(self._work_dir,
                                                           self.__transcription_local_folder_path, "*.json")
        for transcription_file_path in transcription_paths:
            self.logger.debug('Processing %s', transcription_file_path)

            transcription_items = self.__get_transcription_items(transcription_file_path)

            transcription_file_relative_path = transcription_file_path.replace(self._work_dir + '/', '')
            self.__handle_transcription_items(transcription_items, syntax_tokens, transcription_file_relative_path,
                                              metadata_file_path, metadata_file_columns)

    def __get_syntax_tokens(self, local_folder_path):
        syntax_tokens = []

        syntax_paths = io_utils.get_files_in_folder(self._work_dir, local_folder_path, "*.csv")
        for syntax_file_local_path in syntax_paths:
            with open(syntax_file_local_path, 'r') as syntax_file:
                reader = csv.reader(syntax_file)
                next(reader, None)
                for syntax_entry in reader:
                    syntax_tokens.append(syntax_entry[0])

        return syntax_tokens

    @staticmethod
    def __get_transcription_items(transcription_file_path):
        with open(transcription_file_path, 'r') as transcription_file:
            transcription_data = json.load(transcription_file)
            transcription_items = [transcription_item for transcription_item in transcription_data['results']['items']
                                   if transcription_item['type'] == 'pronunciation']

        return transcription_items

    def __create_metadata_file(self, file_columns):
        metadata_file_path = os.path.join(self.__syntax_local_folder_path, self.__metadata_file_name)
        with open(metadata_file_path, 'w') as metadata_file:
            wr = csv.DictWriter(metadata_file, fieldnames=file_columns)
            wr.writeheader()

        return os.path.join(self._work_dir, metadata_file_path)

    def __handle_transcription_items(self, transcription_items, syntax_tokens, transcription_file_path,
                                     metadata_file_path, file_columns):

        with open(metadata_file_path, 'a') as metadata_file:
            wr = csv.DictWriter(metadata_file, fieldnames=file_columns)

            for transcription_item in transcription_items:
                content = transcription_item['alternatives'][0]['content']
                if content in syntax_tokens:
                    self.logger.debug('Found a match: %s', content)
                    video_local_path = io_utils.change_extension(transcription_file_path, '.mp4')
                    video_path = video_local_path.replace('transcriptions', 'not-annotated-videos')

                    wr.writerow({
                        'token': content,
                        'video_path': video_path,
                        'start_time': transcription_item['start_time'],
                        'end_time': transcription_item['end_time']
                    })
