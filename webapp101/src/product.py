import os
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/product/<id>', methods=['GET'])
def product(id):
    return '''
    <!doctype html>
    <title>Product details</title>
    <body>Product</body>
    '''
