from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('heart_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture all 11 inputs from the form
        data = {
            'age': [float(request.form['age'])],
            'gender': [int(request.form['gender'])], # 1: women, 2: men
            'height': [float(request.form['height'])],
            'weight': [float(request.form['weight'])],
            'ap_hi': [float(request.form['ap_hi'])], # Systolic BP
            'ap_lo': [float(request.form['ap_lo'])], # Diastolic BP
            'cholesterol': [int(request.form['cholesterol'])], # 1: normal, 2: above, 3: well above
            'gluc': [int(request.form['gluc'])],
            'smoke': [int(request.form['smoke'])], # 0: no, 1: yes
            'alco': [int(request.form['alco'])],
            'active': [int(request.form['active'])]
        }

        # Medical Sanity Check
        if data['ap_hi'][0] < 70 or data['weight'][0] < 30:
            return render_template('result.html', prediction_text="Unrealistic medical data entered.", alert_class="warning")

        df_input = pd.DataFrame(data)
        
        # Get Probability
        prob = model.predict_proba(df_input)[0][1]
        risk_percent = round(prob * 100, 2)

        if risk_percent >= 50:
            msg = f"High Risk ({risk_percent}% Confidence). Please consult a doctor."
            clr = "danger"
        else:
            msg = f"Low Risk ({risk_percent}% Confidence). Maintain your healthy habits!"
            clr = "success"

        return render_template('result.html', prediction_text=msg, alert_class=clr)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)