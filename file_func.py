import os
import shutil
import pandas as pd
import joblib

# function to create temporary csv data file from it's binary object
def create_file(file_name, data_bytes):
    with open("tmp_files/"+file_name, 'wb') as f:
        f.write(data_bytes)


# delete temporary files
def delete_tmp_files(folder_name):
        folder = folder_name
        for filename in os.listdir(folder):
            
            file_path = os.path.join(folder, filename)
            try:
                os.unlink(file_path)
    
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


# function to perform transformations on the data
def transformations(df, encoder):
    final_data = encoder.fit_transform(data.drop(columns='cost'))
    x = final_data
    y = data['cost']

    return x, y


# function to load the model
def load_models():

    rf_model = joblib.load("models/rf_model.joblib")
    encoder = joblib.load("models/encoder.joblib")


    return rf_model, encoder
