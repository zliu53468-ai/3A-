from flask import Flask, request, jsonify

from flask_cors import CORS

import numpy as np

import joblib

import tensorflow as tf

import os, json



app = Flask(__name__)

CORS(app)



# === 載入模型 ===

dl_model = None

stat_model = None



if os.path.exists("dl_model.h5"):

    dl_model = tf.keras.models.load_model("dl_model.h5")



if os.path.exists("stat_model.pkl"):

    stat_model = joblib.load("stat_model.pkl")



# === 特徵萃取 ===

def extract_features(roadmap):

    features = {

        "banker_streak": 0,

        "player_streak": 0,

        "jump_count": 0,

        "long_banker": 0,

        "long_player": 0

    }

    streak = 0

    last = None

    for item in roadmap:

        if item in ["莊", "閒"]:

            if item == last:

                streak += 1

            else:

                streak = 1

            if item == "莊":

                features["banker_streak"] = max(features["banker_streak"], streak)

            else:

                features["player_streak"] = max(features["player_streak"], streak)

            last = item

        elif item == "跳":

            features["jump_count"] += 1

        elif item == "長莊":

            features["long_banker"] += 1

        elif item == "長閒":

            features["long_player"] += 1

    return np.array(list(features.values())).reshape(1, -1)



# === Big Road 分析 (規則型) ===

def big_road_analysis(roadmap):

    if roadmap.count("莊") > roadmap.count("閒"):

        return np.array([0.6, 0.3, 0.1])

    else:

        return np.array([0.4, 0.5, 0.1])



# === API: /predict ===

@app.route("/predict", methods=["POST"])

def predict():

    data = request.get_json()

    roadmap = data.get("roadmap", [])

    features = extract_features(roadmap)



    preds = []



    # Deep Learning AI

    if dl_model:

        dl_pred = dl_model.predict(features, verbose=0)[0]

        preds.append(dl_pred)



    # Statistical AI

    if stat_model:

        stat_pred = stat_model.predict_proba(features)[0]

        preds.append(stat_pred)



    # Big Road AI

    br_pred = big_road_analysis(roadmap)

    preds.append(br_pred)



    # Multi-Source Fusion = 平均

    final_pred = np.mean(preds, axis=0)



    return jsonify({

        "banker": round(float(final_pred[0]), 2),

        "player": round(float(final_pred[1]), 2),

        "tie": round(float(final_pred[2]), 2)

    })



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)
