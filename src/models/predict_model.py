import pickle

def load_model(model_path):
    """
    Laods a model
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, X):
    """
    Predict with the moedl
    """
    return model.predict(X)

def predict_proba(model, X):
    """
    Predicts with probability
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        raise AttributeError("Le modèle ne supporte pas predict_proba")