# import json
from audio.audio_processor import AudioProcessor


def handler(event, context):
    payload = event['payload']
    operation = event['operation']

    operations = {
        'extract-audio': lambda param: AudioProcessor().process(**param)
    }

    if operation in operations:
        return operations[operation](payload)
    else:
        raise ValueError('Unrecognized operation "{}"'.format(operation))

    # return {
    #     "statusCode": 200,
    #     "body": json.dumps({
    #         "message": "hello world",
    #         # "location": ip.text.replace("\n", "")
    #     }),
    # }
