import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

session = sagemaker.Session()
role = 'arn:aws:iam::427684478694:role/SageMakerRole'

# Create model
model = TensorFlowModel(model_data='s3://sagemaker-us-east-1-427684478694/tensorflow-training-2025-09-13-08-33-30-427/output/model.tar.gz',  
                        role=role,
                        framework_version='2.12.0',
                        entry_point='src/inference.py',
                        source_dir='src',
                        sagemaker_session=session)
# Deploy model to endpoint
predictor = model.deploy(initial_instance_count=1, instance_type='ml.t2.medium',
                         serializer=JSONSerializer(),
                         deserializer=JSONDeserializer())

print(f"Model deployed at endpoint: {predictor.endpoint_name}")

# Example prediction
response = predictor.predict({})
print("Prediction response:", response)