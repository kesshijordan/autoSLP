from flask import Flask

# UPLOAD_FOLDER = '/path/to/the/uploads'
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'tmp/'
app.config['UPLOAD_FOLDER'] = 'islp/static/images/'

# we want to keep this at bottom to avoid circular imports
from islp import views
