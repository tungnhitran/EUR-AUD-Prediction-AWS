import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model('model/1')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    recent_rates = np.array(body['recent_rates']).reshape(1, 30, 1)  # Example input shape
    prediction = model.predict(recent_rates)[0][0]
    return {'statusCode': 200, 'body': json.dumps({'predicted_rate': prediction})}