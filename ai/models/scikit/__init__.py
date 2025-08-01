import joblib
import os

def load_sklearn_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), f'{model_name}.pkl')
    return joblib.load(model_path)