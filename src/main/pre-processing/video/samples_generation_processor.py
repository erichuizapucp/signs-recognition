import logging
import os
import csv

from processor import Processor


class SamplesGenerationProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        syntax_local_folder_path = data['syntax']['localFolderPath']
        syntax_s3_folder_path = data['syntax']['s3Path']

        transcription_local_folder_path = data['transcription']['localFolderPath']
        transcription_s3_folder_path = data['transcription']['s3Path']

        self.__download_syntax_files(syntax_s3_folder_path)

        for syntax_file in os.listdir(syntax_local_folder_path):
            syntax_file_local_path = os.path.join(self._work_dir, syntax_local_folder_path, syntax_file)
            self.__process_syntax_file(syntax_file_local_path)

    def __download_syntax_files(self, s3_folder_path):
        detected_syntax_files = self._s3.list_objects_v2(
            Bucket=self._bucketName,
            Prefix=s3_folder_path)

        for syntax_file in detected_syntax_files['Contents']:
            syntax_file_key = syntax_file['Key']

            if not syntax_file_key.endswith('.csv'):
                continue

            syntax_file_local_path = os.path.join(self._work_dir, syntax_file_key)
            self._s3.download_file(self._bucketName, syntax_file_key, syntax_file_local_path)

            self.logger.debug('%s was downloaded to %s', syntax_file_key, syntax_file_local_path)

    def __process_syntax_file(self, syntax_file_local_path):
        with open(syntax_file_local_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)

            for syntax_entry in reader:
                print(syntax_entry)
