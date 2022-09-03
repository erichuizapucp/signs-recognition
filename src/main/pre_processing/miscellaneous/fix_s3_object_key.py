import boto3

bucket_name = 'erichuiza-bucket'
prefix = 'psl-corpus/'

s3 = boto3.client('s3')
video_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

for video_object in video_objects['Contents']:
    original_video_key: str = video_object['Key']
    if 'ó' in original_video_key or 'á' in original_video_key:
        new_video_key = original_video_key.replace('ó', 'o').replace('á', 'a')

        s3.copy_object(Bucket=bucket_name, CopySource=bucket_name + '/' + original_video_key, Key=new_video_key)
        s3.delete_object(Bucket=bucket_name, Key=original_video_key)


