import boto3
import os
import sagemaker as sage

from sagemaker.estimator import Estimator

data_location = os.getenv('DATA_LOCATION')
instance_type = os.getenv('INSTANCE_TYPE')
checkpoint_location = os.getenv('CHECKPOINT_LOCATION')
checkpoint_local_location = '/opt/ml/checkpoints'

sess = sage.Session()

client = boto3.client('sts')  # AWS Security Token Service
account = client.get_caller_identity()['Account']

my_session = boto3.session.Session()
region = my_session.region_name

algorithm_name = 'self-supervised-psl-training-swav'
ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)

role = os.getenv('SAGEMAKER_ROLE')

hyper_parameters = {
    'mirrored_training': 'True',
    'batch_size': '4',
    'no_replicas': '4',
    'no_epochs': '20',
    'no_steps': '1000',
    'start_learning_rate': '0.1',
    'end_learning_rate': '0.0001',
    'no_projection_1_neurons': '256',
    'no_projection_2_neurons': '128',
    'prototype_vector_dim': '64',
    'lstm_cells': '512',
    'embedding_size': '512'
}
estimator = Estimator(
    role=role,
    instance_count=1,
    instance_type=instance_type,
    image_uri=ecr_image,
    hyperparameters=hyper_parameters,
    metric_definitions=[
        {
            'Name': 'train:EpochCrossEntropyLoss',
            'Regex': 'epoch_loss: (.*?);',
        },
        {
            'Name': 'train:StepCrossEntropyLoss',
            'Regex': 'step_loss: (.*?);',
        }
    ],
    checkpoint_s3_uri=checkpoint_location,
    checkpoint_local_path=checkpoint_local_location,
)
estimator.fit(data_location, wait=False)
