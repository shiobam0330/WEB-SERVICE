# Importamos las bibliotecas necesarias
import uvicorn
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
import tempfile
import shutil  
from pycaret.regression import predict_model
# Crear una instancia de FastAPI
app = FastAPI()

path = "D:/Downloads/INTELIGENCIA ARTIFICIAL/ACTIVIDAD VALENTINA hechaaa/WEB SERVICE/"
prueba = pd.read_csv(path + "prueba_APP.csv",header = 0,sep=";",decimal=",")

with open(path +'best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

covariables = ['Avg. Session Length','Time on App',
                 'Time on Website', 'Length of Membership','dominio', 'Tec']


@app.post("/upload-excel")
def upload_excel(file: UploadFile = File(...)):
    try:
        # Crear un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            # Leer el archivo Excel usando pandas y almacenarlo en un DataFrame
            df = pd.read_excel(temp_file.name)

            base_modelo = df.get(covariables)
            predictions = predict_model(modelo, data=base_modelo)
            predictions["Price"] = predictions["prediction_label"].map(float)
            prediction_label = list(predictions["Price"])

            return {"predictions": prediction_label}

    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}

# Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
