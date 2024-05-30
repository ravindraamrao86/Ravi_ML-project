# src/utils.py

import pickle
import os

def save_object(obj, file_path):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise e
