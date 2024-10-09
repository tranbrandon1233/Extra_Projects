import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_medical_data(filename):
  """Loads medical examination data from a JSON file.

  Args:
    filename: Path to the JSON file containing the medical record.

  Returns:
    A pandas DataFrame containing the medical data.
  """

  with open(filename, 'r') as f:
    data = json.load(f)
  return pd.DataFrame(data)


def preprocess_data(df):
  """Preprocesses the medical data for model training.

  Args:
    df: pandas DataFrame containing the medical data.

  Returns:
    A preprocessed DataFrame with one-hot encoded categorical features and
    numerical features scaled to a common range.
  """

  # Select relevant features
  selected_features = [
      'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
      'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
      'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels',
      'thalassemia'
  ]
  X = df[selected_features]

  # One-hot encode categorical features
  categorical_cols = ['sex', 'chest_pain_type', 'rest_ecg', 'exercise_induced_angina', 
                      'st_slope', 'thalassemia']
  X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

  # Scale numerical features
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  numerical_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate',
                  'st_depression', 'num_major_vessels']
  X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

  # Target variable
  y = df['heart_disease']

  return X, y


def train_model(X, y):
  """Trains a Random Forest model on the medical data.

  Args:
    X: Preprocessed features as a pandas DataFrame.
    y: Target variable as a pandas Series.

  Returns:
    A trained Random Forest Classifier.
  """

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
  return model


def predict_disease(model, data):
  """Predicts the disease based on the trained model and input data.

  Args:
    model: Trained Random Forest Classifier.
    data: A dictionary or list representing a medical examination record (must match the 
          features used for training).

  Returns:
    A string indicating the predicted disease ("Heart Disease" or "No Heart Disease").
  """

  # Preprocess the input data
  input_df = pd.DataFrame([data])
  preprocessed_data = preprocess_data(input_df)[0]

  # Get prediction
  prediction = model.predict(preprocessed_data)[0]

  if prediction == 1:
    return "Heart Disease"
  else:
    return "No Heart Disease"


def main(input_file):
  """Loads, preprocesses, trains a model on medical data, and predicts disease for 
     a given input record.

  Args:
    input_file: Path to the JSON file containing the medical record.
  """

  # Load medical data
  data = load_medical_data(input_file)

  # Train model (using UCI Heart Disease dataset as an example)
  uci_data = pd.read_csv("heart.csv")
  X_uci, y_uci = preprocess_data(uci_data)
  model = train_model(X_uci, y_uci)

  # Predict disease for the input record
  predicted_disease = predict_disease(model, data.iloc[0].to_dict())
  print(f"Predicted Disease: {predicted_disease}")


if __name__ == "__main__":
  input_filename = "data.json"  # Replace with your input file
  main(input_filename)