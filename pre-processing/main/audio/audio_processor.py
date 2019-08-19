import os
from moviepy.editor import VideoFileClip
from downloader import Downloader


class AudioProcessor(Downloader):
    __audio_clips_base = 'audio-clips'
    __audio_clip_extension = '.wav'

    def __init__(self):
        super().__init__()

    def process(self, data):
        super().process(data)

        # extract the audio from video
        audio_file_path = self.__generate_audio_clip_file_path(self._video_file_path)
        if not os.path.exists(audio_file_path):
            os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)

            self.__extract_audio(self._video_file_path, audio_file_path)
            print('the audio clip is available at: {}'.format(audio_file_path))
        else:
            print('the audio clip is already available at: {}'.format(audio_file_path))

        # check if the the audio clip exist in s3
        audio_key = self.__generate_audio_clip_key(audio_file_path)
        self.upload_file(audio_file_path, audio_key)

    @staticmethod
    def __extract_audio(video_file_path, audio_file_path):
        video_clip = VideoFileClip(video_file_path)
        audio_clip = video_clip.audio

        # save to temp file
        audio_clip.write_audiofile(audio_file_path)

    def __generate_audio_clip_file_path(self, video_file_path: str):
        relative_path, _ = os.path.splitext(video_file_path.replace(self._source_videos_base, self.__audio_clips_base))
        return os.path. os.path.join(self._binary_folder, relative_path + self.__audio_clip_extension)

    def __generate_audio_clip_key(self, audio_clip_file_path: str):
        path = audio_clip_file_path.replace(self._binary_folder, '')
        return path[1:] if path.startswith('/') else path
