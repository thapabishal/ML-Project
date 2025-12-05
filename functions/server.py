# File: functions/server.py

# Import the original Flask app instance
from app import app 
# Import the handler from netlify-lambda
from netlify_lambda import handler

# The netlify_lambda library wraps your Flask app and makes it runnable 
# as a serverless function. When Netlify executes this, it calls the handler.
def function_handler(event, context):
    return handler(event, context)

# You may need to rename the wrapper function to be 'server' 
# if the original name causes issues with the redirect target:
# def server(event, context):
#     return handler(event, context)