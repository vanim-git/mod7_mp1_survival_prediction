     
import gradio
import pickle
import pandas as pd
import xgboost

# Load your trained model
survival_model = pickle.load(open('xgboost-model.pkl',"rb"))

# Function for prediction

def predict_death_event(Age, Anaemia, Creatinine_Phosphokinase, Diabetes, Ejection_Fraction, High_Blood_Pressure, Platelets, Serum_Creatinine, Serum_Sodium, Sex, Smoking, Time):
  #x = np.array(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'])
  df1 = pd.DataFrame.from_dict(
        {
            "age": [Age],
            "anaemia": [1 if Anaemia else 0],
            "creatinine_phosphokinase": [Creatinine_Phosphokinase],
            "diabetes": [1 if Diabetes else 0],
            "ejection_fraction": [Ejection_Fraction],
            "high_blood_pressure": [1 if High_Blood_Pressure else 0],
            "platelets": [Platelets],
            "serum_creatinine": [Serum_Creatinine],
            "serum_sodium": [Serum_Sodium],
            "sex": [0 if Sex else 1],
            "smoking": [1 if Smoking else 0],
            "time": [Time],
        }
    )
  
  pred = survival_model.predict_proba(df1)[0]
  #{"Perishes": float(pred[0]), "Survives": float(pred[1])}
  return float(pred[0]), float(pred[1])


title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [
                            gradio.Slider(0, 95, value=40),
                            gradio.Radio(["Yes", "No"], type="index", value=0),
                            gradio.Slider(0, 8000, value = 5000),
                            gradio.Radio(["Yes", "No"], type="index", value = 0),
                            gradio.Slider(0, 100, value=80),
                            gradio.Radio(["Yes", "No"], type="index", value=0),
                            gradio.Slider(0, 900000, value = 225000),
                            gradio.Slider(0.0, 10.0, value = 1.0),
                            gradio.Slider(0, 400, value = 100),
                            gradio.Radio(["Male", "Female"], type="index", value=0),
                            gradio.Radio(["Yes", "No"], type="index", value=0),
                            gradio.Slider(0, 1000, value = 100),
                            ],
                         outputs = [gradio.Number(label="Probablity of death"), gradio.Number(label="Probablity of survival")],
                         title = title,
                         description = description)

iface.launch(server_name = "0.0.0.0", server_port = 8001) 

