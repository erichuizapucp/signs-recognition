import shlex
import time
import os

from subprocess import check_call
from datetime import datetime

from downloader import Downloader
import io_utils


class VideoSplitter(Downloader):
    def __init__(self):
        super().__init__()

    def process(self, data):
        exec_start = time.time()

        super().process(data)
        seqs = data['sequences']

        for idx, seq in enumerate(seqs):
            self.__split_video(seq['start'], seq['end'], idx)

        exec_end = time.time()

        print('Video splitter completed processing {} video chunks in {} seconds'.
              format(len(seqs), round(exec_end - exec_start, 2)))

    def __split_video(self, start, end, idx):
        start_time = datetime.strptime(start, '%M:%S')
        end_time = datetime.strptime(end, '%M:%S')

        duration = \
            datetime.combine(datetime.today(), end_time.time()) - datetime.combine(datetime.today(), start_time.time())

        chunk_path = io_utils.get_video_chunk_path(self._video_file_path, idx)
        io_utils.check_path_dir(chunk_path)

        cmd = "ffmpeg -i {} -ss {} -t {} {} -loglevel panic -y".format(
            self._video_file_path, start_time.strftime('%H:%M:%S'), str(duration), chunk_path)

        if check_call(shlex.split(cmd), universal_newlines=True) == 0:
            print('{} was created successfully'.format(chunk_path))

            chunk_key = io_utils.get_video_chunk_path(self._video_key, idx)
            self.upload_file(chunk_path, chunk_key)
            print('{} was uploaded to s3 successfully.'.format(chunk_path))

            os.remove(chunk_path)
            print('{} was removed from local file system successfully.'.format(chunk_path))
