import sagemaker
from sagemaker.tensorflow import TensorFlow
import os
import boto3

session = sagemaker.Session()
role = 'arn:aws:iam::427684478694:role/SageMakerRole'
bucket = 'eur-aud-bucket'

# Define Tensorflow estimator for LSTM
estimator = TensorFlow(entry_point='src/train_lstm.py',
                        role=role,
                        instance_count=1,
                        instance_type='ml.m5.large',
                        framework_version='2.4.1',
                        py_version='py37',
                        script_mode=True,
                        hyperparameters={
                            'epochs': 50,
                            'batch_size': 32,
                            'learning_rate': 0.001
                        },
                        output_path='s3://eur-aud-bucket/output/',
                        sagemaker_session=session
                        )

# Train the model
estimator.fit({'train': 's3://eur-aud-bucket/data/'})
print("Training job started.")