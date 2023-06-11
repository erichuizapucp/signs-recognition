import logging
import json
import os
import boto3
import csv

from pre_processing.common.processor import Processor
from pre_processing.common import io_utils
from collections import Counter


class TranscriptionSyntaxDetectionProcessor(Processor):
    __transcription_files_prefix = "transcriptions/"
    __nouns_file_name = "detected-nouns.csv"
    __numbers_file_name = "detected-numbers.csv"
    __preferred_language_code = 'es'

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self._comprehend = boto3.client('comprehend')

    def process(self, data):
        total_nouns = []
        total_nums = []

        transcription_folders = self.s3.list_objects_v2(
            Bucket=self.bucketName,
            StartAfter=self.__transcription_files_prefix)

        for transcription_dict in transcription_folders['Contents']:
            transcription_key = transcription_dict['Key']
            if not transcription_key.endswith('.json'):
                continue

            self.logger.debug('Extracting audio video_transcription from: %s', transcription_key)

            file_obj = self.s3.get_object(Bucket=self.bucketName, Key=transcription_key)
            file_content = file_obj['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)

            transcriptions = json_content['results']['transcripts']
            for index, transcription in enumerate(transcriptions):
                transcript = transcription['transcript']
                if len(transcript.encode('utf-8')) > 5000:
                    self.logger.error("The transcript %d in %s cannot be processed because its size is over 5000 bytes",
                                      index, transcription_key)
                    continue

                nouns, nums = self.__handle_syntax(transcript)
                total_nouns.extend(nouns)
                total_nums.extend(nums)

        total_nouns = sorted(dict(Counter(total_nouns)).items(), key=lambda kv: kv[1], reverse=True)
        total_nums = sorted(dict(Counter(total_nums)).items(), key=lambda kv: kv[1], reverse=True)

        local_path = os.path.join(self.work_dir, data['localPath'])
        s3_path = data['s3Path']

        self.__handle_syntax_file_writing(local_path, s3_path, self.__nouns_file_name, total_nouns)
        self.__handle_syntax_file_writing(local_path, s3_path, self.__numbers_file_name, total_nums)

    def __check_syntax_file(self, local_path, s3_path, syntax_file_name):
        local_file_path = os.path.join(local_path, syntax_file_name)
        s3_key = os.path.join(s3_path, syntax_file_name)

        if self.s3_exist_object(s3_key):
            self.logger.debug('audio video_transcription video_transcription-syntax-detection already available in s3 at: %s',
                              s3_key)
            self.s3_delete_object(s3_key)
            self.logger.debug('audio video_transcription %s deleted', s3_key)

        if os.path.exists(local_file_path):
            self.logger.debug('local file %s exist', local_file_path)
            os.remove(local_file_path)
            self.logger.debug('local file %s was deleted', local_file_path)

        io_utils.check_path_file(local_file_path)

        return local_file_path, s3_key

    def __handle_syntax(self, transcription):
        resp = self._comprehend.detect_syntax(
            Text=transcription,
            LanguageCode=self.__preferred_language_code
        )

        nouns = []
        nums = []
        for token in resp['SyntaxTokens']:
            token_tag = token['PartOfSpeech']['Tag']
            if token_tag not in ('NOUN', 'NUM'):
                continue

            token_score = float(token['PartOfSpeech']['Score'])
            if token_score < 0.8:
                continue

            token_text = token['Text']
            nouns.append(token_text) if token_tag == 'NOUN' else nums.append(token_text)

        return nouns, nums

    def __handle_syntax_file_writing(self, local_path, s3_path, syntax_file_name, detected_tokens):
        local_file_path, s3_key = self.__check_syntax_file(local_path, s3_path, syntax_file_name)

        with open(local_file_path, 'w') as f:
            field_names = ['token', 'count']
            wr = csv.DictWriter(f, fieldnames=field_names)
            wr.writeheader()

            for detected_token in detected_tokens:
                wr.writerow({'token': detected_token[0], 'count': detected_token[1]})

        self.logger.debug('audio video_transcription syntax is available locally at: %s',
                          local_file_path)

        self.upload_file(local_file_path, s3_key)
        self.logger.debug('audio video_transcription syntax uploaded to: %s', s3_key)
