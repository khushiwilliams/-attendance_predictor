from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. TRAINING DATA  (dummy dataset, baked into code)
# Each row: [classes_held, classes_attended, study_hours, health (0=Good,1=Avg,2=Poor)]
# Label: 0=Low (<60%), 1=Medium (60-80%), 2=High (>80%)
# ─────────────────────────────────────────────
X_train = np.array([
    [40, 38, 20, 0],  # High
    [40, 36, 18, 0],  # High
    [40, 34, 15, 0],  # High
    [40, 32, 14, 1],  # High
    [40, 30, 12, 0],  # Medium
    [40, 28, 10, 1],  # Medium
    [40, 26,  9, 1],  # Medium
    [40, 24,  8, 1],  # Medium
    [40, 22,  7, 2],  # Low
    [40, 18,  5, 2],  # Low
    [40, 14,  4, 2],  # Low
    [40, 10,  3, 2],  # Low
    [50, 47, 22, 0],  # High
    [50, 43, 18, 0],  # High
    [50, 38, 14, 1],  # Medium
    [50, 33, 10, 1],  # Medium
    [50, 25,  6, 2],  # Low
    [50, 20,  4, 2],  # Low
    [30, 28, 18, 0],  # High
    [30, 24, 12, 0],  # High
    [30, 20,  9, 1],  # Medium
    [30, 16,  6, 2],  # Low
    [60, 57, 25, 0],  # High
    [60, 50, 18, 1],  # Medium
    [60, 42, 12, 1],  # Medium
    [60, 35,  8, 2],  # Low
    [45, 42, 20, 0],  # High
    [45, 38, 15, 0],  # High
    [45, 32, 10, 1],  # Medium
    [45, 25,  6, 2],  # Low
])

y_train = np.array([
    2, 2, 2, 2,   # High
    1, 1, 1, 1,   # Medium
    0, 0, 0, 0,   # Low
    2, 2,         # High
    1, 1,         # Medium
    0, 0,         # Low
    2, 2,         # High
    1,            # Medium
    0,            # Low
    2,            # High
    1, 1,         # Medium
    0,            # Low
    2, 2,         # High
    1,            # Medium
    0,            # Low
])

# ─────────────────────────────────────────────
# 2. TRAIN THE MODEL (happens once at startup)
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

HEALTH_MAP   = {"Good": 0, "Average": 1, "Poor": 2}
LABEL_MAP    = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLOR  = {0: "low", 1: "medium", 2: "high"}
LABEL_EMOJI  = {0: "⚠️", 1: "📊", 2: "🏆"}

# ─────────────────────────────────────────────
# 3. ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- grab form values ---
    name          = request.form.get("name", "Student")
    classes_held  = int(request.form.get("classes_held"))
    classes_att   = int(request.form.get("classes_attended"))
    study_hours   = float(request.form.get("study_hours"))
    health        = request.form.get("health")

    # --- derive attendance % ---
    attendance_pct = round((classes_att / classes_held) * 100, 1)

    # --- encode health ---
    health_encoded = HEALTH_MAP.get(health, 1)

    # --- build feature vector ---
    features = np.array([[classes_held, classes_att, study_hours, health_encoded]])

    # --- predict ---
    pred_label = model.predict(features)[0]
    probas     = model.predict_proba(features)[0]      # [P(Low), P(Med), P(High)]
    confidence = round(max(probas) * 100, 1)

    label_text  = LABEL_MAP[pred_label]
    label_class = LABEL_COLOR[pred_label]
    emoji       = LABEL_EMOJI[pred_label]

    # --- advice per category ---
    advice = {
        "Low":    "Attendance is critically low. Prioritise attending all upcoming classes and speak to your professor.",
        "Medium": "Attendance is satisfactory but can improve. Try to attend every session and reduce absences.",
        "High":   "Excellent attendance! Keep up the great work and maintain this discipline throughout the semester.",
    }[label_text]

    return render_template(
        "result.html",
        name=name,
        attendance_pct=attendance_pct,
        classes_held=classes_held,
        classes_att=classes_att,
        study_hours=study_hours,
        health=health,
        label=label_text,
        label_class=label_class,
        emoji=emoji,
        confidence=confidence,
        advice=advice,
        prob_low=round(probas[0]*100, 1),
        prob_med=round(probas[1]*100, 1),
        prob_high=round(probas[2]*100, 1),
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
