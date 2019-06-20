import boto3
import os


class TranscriptionProcessor:
    def __init__(self):
        self.__transcribe = boto3.client('transcribe')

    def process(self, data):
        audio_clip_uri = data['uri']

        job_name = self.__generate_job_name(audio_clip_uri)

    @staticmethod
    def __generate_job_name(audio_clip_uri):
        return ''
