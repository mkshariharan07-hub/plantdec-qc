import utils
try:
    model, scaler = utils.load_model_and_scaler()
    mode = utils.get_feature_mode(model)
    print(f"Model loaded. Mode: {mode}, Features: {model.n_features_in_}")
except Exception as e:
    import traceback
    traceback.print_exc()
