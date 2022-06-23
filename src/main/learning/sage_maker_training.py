import os

from sagemaker.estimator import Estimator

role = os.getenv("SAGEMAKER_ROLE")
hyper_parameters = {
    'detect_person': True,
    'mirrored_training': True,
    'batch_size': 1,
    'no_replicas': 2,
    'no_epochs': 20,
}
instance_type = "local"
estimator = Estimator(
    role=role,
    instance_count=1,
    instance_type=instance_type,
    image_uri="self-supervised-psl-training-swav",
    hyperparameters=hyper_parameters,
    metric_definitions=[
        {
            'Name': 'train:CrossEntropyLoss',
            'Regex': 'epoch_loss: (.*?);',
        },
        {
            'Name': 'train:CrossEntropyLoss',
            'Regex': 'step_loss: (.*?);',
        }
    ]
)

working_folder = os.getenv('WORK_DIR', './')
estimator.fit("file://{}/dataset".format(working_folder))
