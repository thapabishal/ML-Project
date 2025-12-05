# File: functions/server.py

from netlify_lambda import handler
from app import app 

# This function is the entry point for Netlify's serverless environment.
def server(event, context):
    # Pass the event/context and the Flask app instance to the handler
    return handler(event, context, app)