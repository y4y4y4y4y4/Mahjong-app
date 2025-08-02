from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return "Mahjong AI Flask backend is running!"

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok"})
