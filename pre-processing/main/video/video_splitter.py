import shlex
import os
import ntpath
import time

from subprocess import check_call
from datetime import datetime
from processor import Processor


class VideoSplitter(Processor):
    def __init__(self):
        super().__init__()

    def process(self, data):
        exec_start = time.time()

        super().process(data)
        seqs = data['sequences']

        for idx, seq in enumerate(seqs):
            self.__split_video(seq['start'], seq['end'], idx)

        exec_end = time.time()

        print('Video splittrer completed completed processing {} video chunks in {} seconds'.
              format(len(seqs), round(exec_end - exec_start, 2)))

    def __split_video(self, start, end, idx):
        start_time = datetime.strptime(start, '%M:%S')
        end_time = datetime.strptime(end, '%M:%S')

        duration = \
            datetime.combine(datetime.today(), end_time.time()) - datetime.combine(datetime.today(), start_time.time())

        base_dir = os.path.dirname(self._video_file_path)
        chunk_path = os.path.join(base_dir, 'chunks', str(idx) + '-' + ntpath.basename(self._video_file_path))

        if not os.path.exists(os.path.dirname(chunk_path)):
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

        cmd = "ffmpeg -i {} -ss {} -t {} {} -loglevel panic -y".format(
            self._video_file_path, start_time.strftime('%H:%M:%S'), str(duration), chunk_path)

        if check_call(shlex.split(cmd), universal_newlines=True) == 0:
            print('{} was created successfully'.format(chunk_path))
