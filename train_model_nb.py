from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def train():
    try:
        df = pd.read_csv('./data/WineQT.csv')
    
    except FileNotFoundError:
        print("Error : WIneQT.csv tidak di temukan")

    # independent & dependent
    feature_names = df.columns[0:11].tolist()
    x = df.iloc[:, 0:11].values
    y = df.iloc[:, 11].values


    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # training model

    model = GaussianNB()
    model.fit(x_train, y_train)
    # simpan model

    joblib.dump(model, 'models/wine_model_naive_bayes.pkl')
    print("SUKSES! Model di simpan sebagai 'wine_model_naive_bayes.pkl'")
    return model, feature_names, x_test, y_test

# akurasi
def akurasi(model, x_test, y_test):
    if model is None:
        return
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_predict)
    report_data = classification_report(y_test, y_predict, zero_division=0, output_dict=True)
    f1_weighted = report_data['weighted avg']['f1-score']
    precision = report_data['weighted avg']['precision']
    print(f"AKurasi Model : {accuracy * 100:.2f}%")
    metrics = {'accuracy': accuracy, 'f1_score' : f1_weighted, 'precision' : precision} 
    joblib.dump(metrics, 'models/metrics_nb.pkl')
    print(f"akurasi telah tersimpan di models.metrics_nb.pkl")


# visualisasi
def visualisasi(model, x_test, y_test):
    if model is None:
        print("Model belum di buat, jalankan training dulu.")
    y_predict = model.predict(x_test)
    fig, ax =plt.subplots(figsize=(10,8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_predict, ax=ax, cmap='Blues')
    output_path_png = 'reports/naive_bayes.png'
    output_path_pdf = 'reports/naive_bayes.pdf'
    fig.savefig(output_path_png, bbox_inches='tight')
    fig.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Gambar decision tree berhasil di simpan di : {output_path_png} & {output_path_pdf}")
    plt.close(fig)
    



if __name__ == "__main__":
    model, feature_names, x_test, y_test = train()
    akurasi(model, x_test, y_test)
    visualisasi(model, x_test, y_test)




