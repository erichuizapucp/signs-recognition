import os
import ntpath

from pathlib import Path

chunks_folder = 'chunks'
incoming_queue_folder = 'incoming-queue'
binary_folder = 'binary'


def get_filename_without_extension(file_path):
    return os.path.splitext(ntpath.basename(file_path))[0]


def get_filename(file_path):
    return os.path.splitext(ntpath.basename(file_path))[0]


def get_file_extension(file_path, exclude_dot=False):
    extension = os.path.splitext(ntpath.basename(file_path))[1]
    return extension.replace('.', '') if exclude_dot else extension


def add_prefix_to_filename(file_path, prefix, separator='-'):
    name_without_ext = get_filename_without_extension(file_path)
    ext = get_file_extension(file_path)
    return name_without_ext + separator + prefix + ext


def get_video_chunk_path(video_path, idx):
    base_dir = os.path.dirname(video_path)
    return os.path.join(base_dir, chunks_folder, add_prefix_to_filename(video_path, str(idx).zfill(2)))


def get_video_chunk_base_key(video_key):
    return os.path.join(os.path.dirname(video_key), chunks_folder)


def check_path_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


def change_extension(file_path, new_extension):
    pre, ext = os.path.splitext(file_path)
    return pre + new_extension


def get_files_in_folder(root_path, local_folder_path, extension):
    return [os.path.join(root_path, str(file_path)) for file_path in list(Path(local_folder_path).rglob(extension))]
