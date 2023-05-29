import boto3
from pre_processing.common import io_utils

bucket_name = 'erichuiza-bucket'
prefix = 'short-serialized-psl-corpus/'

s3 = boto3.client('s3')
file_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
extension = '.tfrecord'

for file_object in file_objects['Contents']:
    original_file_key: str = file_object['Key']

    if io_utils.get_file_extension(original_file_key) == extension:
        file_name: str = io_utils.get_filename_without_extension(original_file_key)
        suffix: str = file_name.split('_')[1]
        new_file_key = prefix + 'swav_' + suffix.zfill(5) + extension

        s3.copy_object(Bucket=bucket_name, CopySource=bucket_name + '/' + original_file_key, Key=new_file_key)
        s3.delete_object(Bucket=bucket_name, Key=original_file_key)
