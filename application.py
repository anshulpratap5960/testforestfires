from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "POST":

        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        BUI = float(request.form.get("BUI"))
        Region = float(request.form.get("Region"))

        # 🔥 IMPORTANT: ORDER MUST MATCH TRAINING
        features = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, BUI, Region]]

        scaled_data = standard_scaler.transform(features)

        prediction = ridge_model.predict(scaled_data)[0]

        result = round(prediction, 2)

        return render_template("home.html", result=result)

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)