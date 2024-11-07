from flask import Flask
# from flask_cors import CORS

app = Flask(__name__)

# Import routes to register them with the app
from app import api_routes