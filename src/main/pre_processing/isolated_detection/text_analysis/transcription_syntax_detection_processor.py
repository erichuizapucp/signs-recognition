import logging
import json
import os
import boto3
import csv

from pre_processing.common.processor import Processor
from pre_processing.common import io_utils
from collections import Counter


class TranscriptionSyntaxDetectionProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.comprehend = boto3.client('comprehend')

        self.transcription_files_prefix = "transcriptions/"
        self.excluded_folder = "old_transcriptions/"
        self.nouns_file_name = "detected-nouns.csv"
        self.numbers_file_name = "detected-numbers.csv"
        self.preferred_language_code = 'es'

        self.transcription_batch_max_size = 25
        self.single_syntax_detection_bytes_limit = 4000

    def process(self, data):
        total_nouns = []
        total_nums = []

        transcription_folders = self.s3.list_objects_v2(
            Bucket=self.bucketName,
            Prefix=self.transcription_files_prefix)

        for transcription_dict in transcription_folders['Contents']:
            transcription_key = transcription_dict['Key']

            if self.excluded_folder in transcription_key:
                continue

            if not transcription_key.endswith('.json'):
                continue

            self.logger.debug('Extracting audio video_transcription from: %s', transcription_key)

            file_obj = self.s3.get_object(Bucket=self.bucketName, Key=transcription_key)
            file_content = file_obj['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)

            transcriptions = json_content['results']['transcripts']
            for index, transcription in enumerate(transcriptions):
                transcript = transcription['transcript']
                encoded_transcript = transcript.encode('utf-8')

                is_batched = len(encoded_transcript) > self.single_syntax_detection_bytes_limit
                nouns, nums = self.batch_detect_nouns_nums(transcript) if is_batched else self.single_detect_nouns_nums(transcript)
                total_nouns.extend(nouns)
                total_nums.extend(nums)

        total_nouns = sorted(dict(Counter(total_nouns)).items(), key=lambda kv: kv[1], reverse=True)
        total_nums = sorted(dict(Counter(total_nums)).items(), key=lambda kv: kv[1], reverse=True)

        local_path = os.path.join(self.work_dir, data['localPath'])
        s3_path = data['s3Path']

        self.handle_syntax_file_writing(local_path, s3_path, self.nouns_file_name, total_nouns)
        self.handle_syntax_file_writing(local_path, s3_path, self.numbers_file_name, total_nums)

    def check_syntax_file(self, local_path, s3_path, syntax_file_name):
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

    def single_detect_nouns_nums(self, transcript):
        resp = self.comprehend.detect_syntax(
            Text=transcript,
            LanguageCode=self.preferred_language_code
        )

        return self.detect_nouns_nums(resp)

    def batch_detect_nouns_nums(self, transcript):
        transcriptions_batch = self.split_large_transcription(transcript)

        batch_nouns = []
        batch_nums = []

        # for batch_index in range(len(transcriptions_batch)):
        #     batch = transcriptions_batch[batch_index]
        response = self.comprehend.batch_detect_syntax(TextList=transcriptions_batch,
                                                       LanguageCode=self.preferred_language_code)

        for resp in response['ResultList']:
            nouns, nums = self.detect_nouns_nums(resp)

            batch_nouns.extend(nouns)
            batch_nums.extend(nums)

        return batch_nouns, batch_nums

    @staticmethod
    def detect_nouns_nums(resp):
        nouns = []
        nums = []
        for token in resp['SyntaxTokens']:
            token_tag = token['PartOfSpeech']['Tag']
            if token_tag not in ('NOUN', 'NUM'):
                continue

            token_score = float(token['PartOfSpeech']['Score'])
            if token_score < 0.8:
                continue

            token_text = token['Text'].lower()
            nouns.append(token_text) if token_tag == 'NOUN' else nums.append(token_text)

        return nouns, nums

    def handle_syntax_file_writing(self, local_path, s3_path, syntax_file_name, detected_tokens):
        local_file_path, s3_key = self.check_syntax_file(local_path, s3_path, syntax_file_name)

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

    def split_large_transcription(self, large_transcription):
        words = large_transcription.split()
        result = []
        chunk = ''

        for word in words:
            if len(chunk) + len(word) + 1 > self.single_syntax_detection_bytes_limit:
                result.append(chunk.rstrip())
                chunk = word
            else:
                chunk += ' ' + word

        if chunk:
            result.append(chunk)

        return result
