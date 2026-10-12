"""App web para predecir la supervivencia de un pasajero del Titanic.

Uso:
    pip install -r requirements.txt
    (ejecuta antes 5_pipeline.ipynb para generar models/titanic_pipeline.joblib)
    python app.py
    Abrir http://127.0.0.1:5000

También expone una API JSON:
    POST /api/predict  {"Pclass": 1, "Name": "Doe, Mrs. Jane", "Sex": "female", "Age": 30,
                        "SibSp": 1, "Parch": 0, "Fare": 80, "Cabin": "C85", "Embarked": "C"}
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import titanic_features  # noqa: F401  (necesario para cargar el pipeline guardado)

MODEL_PATH = "models/titanic_pipeline.joblib"
model = joblib.load(MODEL_PATH)
app = Flask(__name__)

COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
           "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

PAGE = """
<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicción Titanic</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:560px;margin:32px auto;padding:0 16px;background:#f6f7f9;color:#1d2433}
 form{background:#fff;padding:20px;border-radius:10px;box-shadow:0 1px 4px #0001;display:grid;gap:10px}
 label{display:grid;gap:4px;font-size:14px} input,select{padding:8px;border:1px solid #ccd;border-radius:6px}
 button{padding:10px;border:0;border-radius:6px;background:#2457c5;color:#fff;font-size:15px;cursor:pointer}
 .res{margin-top:16px;padding:16px;border-radius:10px;background:#fff;box-shadow:0 1px 4px #0001}
 .ok{color:#127a3a} .no{color:#b42318}
</style></head><body>
<h1>¿Sobreviviría al Titanic?</h1>
<form method="post">
 <label>Título <select name="Title">
  {% for t in ["Mr","Mrs","Miss","Master","Dr","Rev"] %}<option {{'selected' if f.get('Title')==t}}>{{t}}</option>{% endfor %}
 </select></label>
 <label>Sexo <select name="Sex"><option value="male" {{'selected' if f.get('Sex')=='male'}}>Hombre</option>
  <option value="female" {{'selected' if f.get('Sex')=='female'}}>Mujer</option></select></label>
 <label>Clase <select name="Pclass">{% for c in [1,2,3] %}<option {{'selected' if f.get('Pclass')==c|string}}>{{c}}</option>{% endfor %}</select></label>
 <label>Edad <input type="number" step="0.5" min="0" max="100" name="Age" value="{{f.get('Age','30')}}"></label>
 <label>Hermanos / cónyuge a bordo <input type="number" min="0" name="SibSp" value="{{f.get('SibSp','0')}}"></label>
 <label>Padres / hijos a bordo <input type="number" min="0" name="Parch" value="{{f.get('Parch','0')}}"></label>
 <label>Tarifa (libras) <input type="number" step="0.01" min="0" name="Fare" value="{{f.get('Fare','15')}}"></label>
 <label>Cabina (opcional, ej. C85) <input name="Cabin" value="{{f.get('Cabin','')}}"></label>
 <label>Puerto <select name="Embarked">
  {% for k,v in [("S","Southampton"),("C","Cherbourg"),("Q","Queenstown")] %}<option value="{{k}}" {{'selected' if f.get('Embarked')==k}}>{{v}}</option>{% endfor %}
 </select></label>
 <button>Predecir</button>
</form>
{% if p is not none %}
<div class="res"><h2 class="{{'ok' if p>=0.5 else 'no'}}">{{'Sobrevive' if p>=0.5 else 'No sobrevive'}}</h2>
<p>Probabilidad estimada de sobrevivir: <b>{{'%.1f'|format(p*100)}} %</b></p></div>
{% endif %}
</body></html>
"""


def to_frame(data: dict) -> pd.DataFrame:
    title = data.get("Title")
    row = {
        "PassengerId": 0,
        "Pclass": int(data.get("Pclass", 3)),
        "Name": data.get("Name") or f"Pasajero, {title or 'Mr'}. Ejemplo",
        "Sex": data.get("Sex", "male"),
        "Age": float(data["Age"]) if data.get("Age") not in (None, "") else None,
        "SibSp": int(data.get("SibSp", 0) or 0),
        "Parch": int(data.get("Parch", 0) or 0),
        "Ticket": data.get("Ticket", "N/A"),
        "Fare": float(data["Fare"]) if data.get("Fare") not in (None, "") else None,
        "Cabin": data.get("Cabin") or None,
        "Embarked": data.get("Embarked", "S"),
    }
    return pd.DataFrame([row], columns=COLUMNS)


@app.route("/", methods=["GET", "POST"])
def index():
    prob = None
    if request.method == "POST":
        prob = float(model.predict_proba(to_frame(request.form))[0, 1])
    return render_template_string(PAGE, p=prob, f=request.form)


@app.post("/api/predict")
def api_predict():
    prob = float(model.predict_proba(to_frame(request.get_json(force=True)))[0, 1])
    return jsonify({"probabilidad_sobrevivir": round(prob, 4),
                    "prediccion": int(prob >= 0.5)})


if __name__ == "__main__":
    app.run(debug=True)
