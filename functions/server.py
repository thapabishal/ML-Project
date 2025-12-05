import awsgi
# Import your Flask app instance
from app import app 


# It receives the AWS Lambda event and context objects.
def handler(event, context):
    # awsgi converts the Lambda event into a standard WSGI request 
    # and converts Flask's response back into a Lambda response.
    return awsgi.response(app, event, context)