from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import joblib
import tensorflow as tf
import os

app = FastAPI(title="CekSawit Rice Prediction API")

# --- GLOBAL VARIABLES ---
lstm_models = []
lgbm_model = None
scaler = None

# --- LOAD MODELS SAAT SERVER NYALA ---
@app.on_event("startup")
def load_artifacts():
    global lstm_models, lgbm_model, scaler
    print("🔄 Loading models & scaler...")
    
    # 1. Load 5 LSTM Models
    for i in range(5):
        path = f"saved_models/lstm_fold_{i}.keras"
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            lstm_models.append(model)
            print(f"   -> Loaded LSTM Fold {i}")
            
    # 2. Load LightGBM
    if os.path.exists("saved_models/lgbm_meta.pkl"):
        lgbm_model = joblib.load("saved_models/lgbm_meta.pkl")
        print("   -> Loaded LightGBM")
        
    # 3. Load Scaler
    if os.path.exists("saved_models/scaler_minmax.pkl"):
        scaler = joblib.load("saved_models/scaler_minmax.pkl")
        print("   -> Loaded Scaler")
    
    print("✅ System Ready!")

# --- DEFINISI INPUT ---
class PredictionRequest(BaseModel):
    # User mengirim list of list (7 baris data)
    # Setiap baris berisi: [Harga_Beras_Asli, Fitur1_Asli, Fitur2_Asli...]
    data_7_hari: list[list[float]]

@app.post("/predict")
def predict_price(payload: PredictionRequest):
    try:
        # 1. Konversi ke Numpy
        input_data = np.array(payload.data_7_hari)
        
        # Validasi bentuk data (harus 7 hari)
        if input_data.shape[0] != 7:
            raise HTTPException(status_code=400, detail="Data harus berisi tepat 7 hari.")
            
        # 2. PREPROCESSING KHUSUS (Sesuai kode normalisasi Anda)
        # Struktur data: Kolom 0 = Harga Beras (Target), Kolom 1 ke belakang = Fitur lain
        
        # Pisahkan Harga (tidak perlu discaling ulang jika input user sudah Rupiah asli)
        # TAPI: Kode training Anda pakai data yang targetnya UN-SCALED, tapi fiturnya SCALED.
        # Jadi kita harus scale fitur-fiturnya saja.
        
        prices_raw = input_data[:, 0].reshape(-1, 1) # Kolom 0
        features_raw = input_data[:, 1:]             # Kolom 1 s/d akhir
        
        # Scale fitur menggunakan scaler yang sudah diload
        if scaler:
            features_scaled = scaler.transform(features_raw)
        else:
            raise HTTPException(status_code=500, detail="Scaler not found on server")
            
        # Gabungkan kembali: [Harga_Asli, Fitur_Scaled]
        # Ini format yang dimakan LSTM saat training tadi
        final_input_sequence = np.hstack([prices_raw, features_scaled])
        
        # Reshape untuk LSTM (1 sampel, 7 timesteps, n fitur)
        lstm_input = final_input_sequence.reshape(1, 7, final_input_sequence.shape[1])
        
        # 3. PREDIKSI LSTM (ENSEMBLE 5 MODEL)
        lstm_preds = []
        for model in lstm_models:
            pred = model.predict(lstm_input, verbose=0)
            lstm_preds.append(pred[0][0])
            
        # Rata-rata hasil 5 LSTM
        avg_lstm_pred = np.mean(lstm_preds)
        
        # 4. PREDIKSI LIGHTGBM
        # Input LightGBM cuma hasil prediksi LSTM
        meta_input = np.array([[avg_lstm_pred]])
        final_price = lgbm_model.predict(meta_input)[0]
        
        return {
            "status": "success",
            "prediksi_harga_besok": float(final_price),
            "satuan": "Rupiah",
            "detail_lstm": float(avg_lstm_pred) # Debugging
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)