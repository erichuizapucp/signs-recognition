import boto3
import os
from moviepy.editor import VideoFileClip
from common.s3_utils import s3_exist_object


class AudioProcessor:
    __source_videos_base = 'not-annotated-videos'
    __audio_clips_base = 'audio-clips'
    __audio_clip_extension = '.wav'

    def __init__(self):
        self.__s3 = boto3.client('s3')
        self.__bucketName = os.environ['S3_BUCKET']
        self.__binary_folder = os.environ['BINARY_FOLDER']

    def process(self, video):
        video_key = video['id']

        # download the video clip from aws s3 and save to temp file
        video_file_path = self.__generate_video_file_path(video_key)
        if not os.path.exists(video_file_path):
            os.makedirs(os.path.dirname(video_file_path), exist_ok=True)

            self.__s3.download_file(self.__bucketName, video_key, video_file_path)
            print('the video clip is available at: {}'.format(video_file_path))
        else:
            print('the video is already available at: {}'.format(video_file_path))

        # extract the audio from video
        audio_file_path = self.__generate_audio_clip_file_path(video_file_path)
        if not os.path.exists(audio_file_path):
            os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)

            self.__extract_audio(video_file_path, audio_file_path)
            print('the audio clip is available at: {}'.format(audio_file_path))
        else:
            print('the audio clip is already available at: {}'.format(audio_file_path))

        # check if the the audio clip exist in s3
        audio_key = self.__generate_audio_clip_key(audio_file_path)
        if not s3_exist_object(self.__s3, self.__bucketName, audio_key):
            # upload the audio clip to s3
            self.__s3.upload_file(audio_file_path, self.__bucketName, audio_key)
            print('the audio clip was uploaded to {}/{}'.format(self.__bucketName, audio_key))
        else:
            print('the audio clip is already available at: {}'.format(audio_file_path))

    @staticmethod
    def __extract_audio(video_file_path, audio_file_path):
        video_clip = VideoFileClip(video_file_path)
        audio_clip = video_clip.audio

        # save to temp file
        audio_clip.write_audiofile(audio_file_path)

    def __generate_video_file_path(self, video_key):
        return os.path.join(self.__binary_folder, video_key)

    def __generate_audio_clip_file_path(self, video_file_path: str):
        relative_path, _ = os.path.splitext(video_file_path.replace(self.__source_videos_base, self.__audio_clips_base))
        return os.path. os.path.join(self.__binary_folder, relative_path + self.__audio_clip_extension)

    def __generate_audio_clip_key(self, audio_clip_file_path: str):
        path = audio_clip_file_path.replace(self.__binary_folder, '')
        return path[1:] if path.startswith('/') else path
