from flask import Flask, request, render_template, jsonify
import numpy as np

from neuralNet import BasicNeuralNet
from graphingNet import GraphingNeuralNet

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    input_data = request.form.get("input_data")
    target_data = request.form.get("target_data")
    epochs = int(request.form.get("epochs"))
    learning_rate = float(request.form.get("learning_rate"))

    input_data = [list(map(float, line.split(','))) for line in input_data.strip().split('\n')]
    target_data = [list(map(float, line.split(','))) for line in target_data.strip().split('\n')]

    nn = GraphingNeuralNet(len(input_data[0]), 4, len(target_data[0]))
    nn.train_epochs(input_data, target_data, epochs, learning_rate)
    predictions = nn.predict(input_data)

    training_error_data = {"x": list(range(len(nn.training_errors))), "y": nn.training_errors, "type": "scatter"}
    output_data = {"x": list(range(len(predictions))), "y": predictions, "type": "scatter"}

    return jsonify({"training_error_chart": training_error_data, "output_chart": output_data, "predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
