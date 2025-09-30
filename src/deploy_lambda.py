import boto3

lambda_client = boto3.client('lambda')
iam_client = boto3.client('iam')

role = iam_client.create_role('LambdaSageMakerRole')

# Upload zip package
with open('lambda_deploy_package.zip', 'rb') as f:
    lambda_client.create_function(
        FunctionName='eur-aud-predictor',
        Runtime='python3.8',
        Role=role['Role']['Arn'],
        Handler='inference.lambda_handler',
        Code={'ZipFile': f.read()},
    )
print("Lambda function created.")