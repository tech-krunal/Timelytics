import pickle
import joblib
import sklearn

print(f"Using scikit-learn version: {sklearn.__version__}")

# Load old model
old_model_path = "app/model/voting_model.pkl"
with open(old_model_path, "rb") as f:
    voting_model = pickle.load(f)

# Save using joblib (more stable than pickle)
new_model_path = "app/model/voting_model_updated.pkl"
joblib.dump(voting_model, new_model_path)

print(f"Model successfully re-saved at: {new_model_path}")
