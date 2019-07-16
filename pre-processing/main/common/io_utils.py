import os
import ntpath

chunks_dir = 'chunks'


def get_filename_without_extension(file_path):
    return os.path.splitext(ntpath.basename(file_path))[0]


def get_file_extension(file_path):
    return os.path.splitext(ntpath.basename(file_path))[1]


def add_prefix_to_filename(file_path, prefix, separator='-'):
    name_without_ext = get_filename_without_extension(file_path)
    ext = get_file_extension(file_path)
    return name_without_ext + separator + prefix + ext


def get_video_chunk_path(video_path, idx):
    base_dir = os.path.dirname(video_path)
    return os.path.join(base_dir, chunks_dir, add_prefix_to_filename(video_path, str(idx)))


def get_video_chunk_base_key(video_key):
    return os.path.join(os.path.dirname(video_key), chunks_dir)


def check_path_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
