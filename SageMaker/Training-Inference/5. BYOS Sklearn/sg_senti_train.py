
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd
import argparse
import os 


vectorizer = None


def model_fn(model_dir):
    """
    Load model created by Sagemaker training.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    load_vectorizer(model_dir)
    return model


def load_vectorizer(model_dir):
    global vectorizer
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    


def input_fn(request_body, request_content_type):
    """
    :param request_body: raw review text
    :param request_content_type: text/csv
    """
    if request_content_type == 'text/csv':
        X = vectorizer.transform([request_body])
        X_nd = X.toarray()
        return X_nd
    else:
        raise ValueError("The model only supports text/csv input")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    return str(prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args = parser.parse_args()
    
    # Load data from the location specified by args.train (In this case, an S3 bucket)
    data = pd.read_csv(os.path.join(args.train, 'train.csv'), index_col=0, engine='python')
    reviews = data['review']
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews)
    features = vectorizer.transform(reviews)
    X_train = pd.DataFrame(features.toarray())
    y_train = data['label']
    logistic_regression_model = LogisticRegression(solver='lbfgs')
    logistic_regression_model.fit(X=X_train, y=y_train)
    # Save the model to the location specified by args.model_dir
    joblib.dump(logistic_regression_model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))
