from flask import Flask, render_template, request
import pickle
import numpy as np
import os  # Needed to safely handle paths

app = Flask(__name__)

# Get absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "titanic_survival_model.pkl")

# Load the trained model and scaler
with open(MODEL_PATH, "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        # Get user input from the form
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        # Create input array
        data = np.array([[pclass, sex, age, fare, embarked]])
        # Scale Age and Fare
        data[:, [2,3]] = scaler.transform(data[:, [2,3]])

        # Predict
        result = model.predict(data)[0]
        prediction = "Survived" if result == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
