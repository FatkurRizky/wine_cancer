import numpy as np
from flask import Flask, render_template, request
from joblib import load
import os

app = Flask(__name__)

# --- KONFIGURASI PATH ---
MODEL_NB_PATH = 'models/wine_model_naive_bayes.pkl'
MODEL_DT_PATH = 'models/wine_model_decision_tree.pkl' 

METRIC_PATH_DT = 'models/metrics_dt.pkl'
METRIC_PATH_NB = 'models/metrics_nb.pkl'

VIZ_IMAGE_NB = 'naive_bayes.png'
VIZ_IMAGE_DT = 'decision_tree.png'

loaded_models = {}

# 1. Load Naive Bayes Model
if os.path.exists(MODEL_NB_PATH):
    loaded_models['naive_bayes'] = load(MODEL_NB_PATH)
    print("Model Naive Bayes berhasil dimuat")
else:
    loaded_models['naive_bayes'] = None

# 2. Load Decision Tree Model
if os.path.exists(MODEL_DT_PATH):
    loaded_models['decision_tree'] = load(MODEL_DT_PATH)
    print("Model Decision Tree berhasil dimuat")
else:
    loaded_models['decision_tree'] = None


# --- FUNGSI BARU: BACA 2 FILE METRICS SEKALIGUS ---
def get_all_metrics():

    results = {
        'nb': {'acc': '0', 'f1': '0'},
        'dt': {'acc': '0', 'f1': '0'}
    }

    def process_file(path):
        data_out = {'acc': '0', 'f1': '0'}
        if os.path.exists(path):
            try:
                data = load(path)
                if isinstance(data, dict):
                    raw_acc = data.get('accuracy', 0)
                    raw_f1  = data.get('f1_score', 0)
                    
                    # Format ke Persen String (0.8512 -> "85.12")
                    data_out['acc'] = f"{float(raw_acc) * 100:.2f}"
                    data_out['f1']  = f"{float(raw_f1) * 100:.2f}"
            except Exception as e:
                print(f"Error baca file {path}: {e}")
        return data_out

    # 1. Proses NB
    results['nb'] = process_file(METRIC_PATH_NB)
    
    # 2. Proses DT
    results['dt'] = process_file(METRIC_PATH_DT)

    return results


@app.route('/')
def home():
    # Panggil fungsi metrics
    m = get_all_metrics()
    
    return render_template('index.html', 
                           # Kirim data NB
                           acc_nb=m['nb']['acc'], 
                           f1_nb=m['nb']['f1'],
                           # Kirim data DT
                           acc_dt=m['dt']['acc'], 
                           f1_dt=m['dt']['f1'],
                           
                           viz_image_nb=VIZ_IMAGE_NB, 
                           viz_image_dt=VIZ_IMAGE_DT)



@app.route('/predict', methods=['POST'])
def predict():
    # Tetap ambil metrics supaya tampilan angka tidak hilang saat submit
    m = get_all_metrics()
    
    # Bungkus variabel template biar rapi
    template_data = {
        'acc_nb': m['nb']['acc'], 'f1_nb': m['nb']['f1'],
        'acc_dt': m['dt']['acc'], 'f1_dt': m['dt']['f1'],
        'viz_image_nb': VIZ_IMAGE_NB, 
        'viz_image_dt': VIZ_IMAGE_DT
    }

    try:
        if request.method == 'POST':
            selected_algo = request.form.get('algorithm')
            model = loaded_models.get(selected_algo)
            
            if model is None:
                return render_template('index.html', 
                                       prediction_text=f"Error: Model {selected_algo} belum siap.",
                                       **template_data) # Unpack data metrics

            # Input Data
            input_data = [
                float(request.form['fa']), float(request.form['va']), float(request.form['ca']), 
                float(request.form['rs']), float(request.form['c']), float(request.form['fsd']), 
                float(request.form['tsd']), float(request.form['d']), float(request.form['ph']), 
                float(request.form['s']), float(request.form['a'])
            ]
            
            # Prediksi
            prediksi = model.predict([input_data])[0]
            algo_display = "Naive Bayes" if selected_algo == 'naive_bayes' else "Decision Tree"
            if selected_algo == 'naive_bayes':
                template_data['model_acc'] = m['nb']['acc']
                template_data['model_f1'] = m['nb']['f1']
            
            else:
                template_data['model_acc'] = m['dt']['acc']
                template_data['model_f1'] = m['dt']['f1']
            return render_template('index.html', 
                                   prediction_text=f"[{algo_display}] Prediksi Kualitas: {prediksi}",
                                   **template_data) # Unpack data metrics lagi
                                   
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f"Terjadi kesalahan input: {e}",
                               **template_data)

if __name__ == "__main__":
    app.run(debug=True)