
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import argparse
import os


# Dictionary to convert labels to indices
LABEL_TO_INDEX = {'Iris-virginica': 0, 'Iris-versicolor': 1, 'Iris-setosa': 2}
# Dictionary to convert indices to labels
INDEX_TO_LABEL = {0: 'Iris-virginica', 1: 'Iris-versicolor', 2: 'Iris-setosa'}


def model_fn(model_dir):
    """
    :param model_dir: (string) specifies location of saved model.
    
    This function is used by AWS Sagemaker to load the model for deployment. 
    
    It does this by simply loading the model that was saved at the end of the 
    __main__ training block above and returning it to be used by the predict_fn
    function below.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    """
    :param request_body: the body of the request sent to the model. The type can vary.
    :param request_content_type: (string) specifies the format/variable type of the request.
    
    This function is used by AWS Sagemaker to format a request body that is sent to 
    the deployed model.
    
    In order to do this, we must transform the request body into a numpy array and
    return that array to be used by the predict_fn function below.
    
    Note: Often times, you will have need to handle other request_content_types. 
    However, in this simple case, we are only going to accept text/csv and raise an error 
    for all other formats.
    """
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float, r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")


def predict_fn(input_data, model):
    """
    :param input_data: (numpy array) returned array from input_fn above. 
    :param model (sklearn model) returned model loaded from model_fn above.
    
    This function is used by AWS Sagemaker to make the prediction on the data
    formatted by the input_fn above using the trained model.
    """
    return model.predict(input_data)


def output_fn(prediction, content_type):
    """
    :param prediction: the returned value from predict_fn above.
    :param content_type: (string) the content type the endpoint expects to be returned.
    
    This function reformats the predictions returned from predict_fn to the final
    format that will be returned as the API call response.
    
    Note: Often times, you will have to handle other request_content_types. 
    """
    print(prediction)
    print('|'.join([INDEX_TO_LABEL[idx] for idx in prediction]))
    return '|'.join([INDEX_TO_LABEL[idx] for idx in prediction])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args = parser.parse_args()
    print(args.output_data_dir)
    print(args.model_dir)
    print(args.train)
    print(args.test)
    
    # Load data from the location specified by args.train (In this case, an S3 bucket)
    data = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")

    # Separate input variables and labels
    train_X = data[[col for col in data.columns if col != 'label']]
    train_Y = data[['label']]

    # Convert labels from text to indices
    train_Y_enc = train_Y['label'].map(LABEL_TO_INDEX)
    print(train_X.head(5))
    print(train_Y_enc.head(5))
    
    # Train the logistic regression model using the fit method
    model = LogisticRegression().fit(train_X, train_Y_enc)
    
    # Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))