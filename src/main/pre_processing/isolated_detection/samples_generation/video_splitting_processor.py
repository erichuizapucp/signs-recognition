import shlex
import time
import os
import logging
import io_utils

from subprocess import check_call
from datetime import datetime

from pre_processing.common.downloader import Downloader


class VideoSplittingProcessor(Downloader):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def process(self, data):
        exec_start = time.time()
        self.logger.debug('Video splitting process started at %s', exec_start)

        super().process(data)
        seqs = data['sequences']

        for index, seq in enumerate(seqs):
            try:
                chunk_path = self.__split_video(seq['start'], seq['end'], index)
                self.logger.debug('%s processed is complete', chunk_path)
            except Exception as e:
                self.logger.error(e)

        exec_end = time.time()
        self.logger.debug('Video splitting completed processing %s samples_generation chunks in %s seconds',
                          len(seqs), round(exec_end - exec_start, 2))

    def __split_video(self, start, end, idx):
        start_time = datetime.strptime(start, '%M:%S')
        end_time = datetime.strptime(end, '%M:%S')

        duration = \
            datetime.combine(datetime.today(), end_time.time()) - datetime.combine(datetime.today(), start_time.time())

        chunk_path = io_utils.get_video_chunk_path(self._video_file_path, idx)
        io_utils.check_path_file(chunk_path)

        cmd = "ffmpeg -i {} -ss {} -t {} {} -loglevel panic -y".format(
            self._video_file_path, start_time.strftime('%H:%M:%S'), str(duration), chunk_path)

        if check_call(shlex.split(cmd), universal_newlines=True) == 0:
            self.logger.debug('%s was created successfully', chunk_path)

            chunk_key = io_utils.get_video_chunk_path(self.video_key, idx)
            self.upload_file(chunk_path, chunk_key)
            self.logger.debug('%s was uploaded to s3 successfully.', chunk_path)

            os.remove(chunk_path)
            self.logger.debug('%s was removed from local file system successfully.', chunk_path)

        return chunk_path
