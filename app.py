from flask import Flask
import sys
import logging
import os
from datetime import datetime
from src.logger import logging
from src.exception import CustomException

app =Flask(__name__)

@app.route('/', methods = ['GET','POST'])

def index():
    try:
        raise Exception('we are testing out custom file')
    except Exception as e:
        abc = CustomException(e,sys)
        logging.info(abc.error_message)

        return "welcome to our Data science project community & project session"

if __name__ == "__main__":
    app.run(debug=True)
    