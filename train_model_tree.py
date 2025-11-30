from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def train():
    try:
        df = pd.read_csv('./data/WineQT.csv')
    
    except FileNotFoundError:
        print("Error : WIneQT.csv tidak di temukan")

    # independent & dependent
    feature_names = df.columns[:-2].tolist()
    x = df.iloc[:, :-2].values
    y = df.iloc[:, -2].values


    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # training model

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    # simpan model

    joblib.dump(model, 'models/wine_model_decision_tree.pkl')
    print("SUKSES! Model di simpan sebagai 'wine_model_decision_tree.pkl'")
    return model, feature_names, x_test, y_test


def akurasi(model, x_test, y_test):
    if model is None:
        return
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_predict)
    report_data = classification_report(y_test, y_predict, output_dict=True)
    f1_weighted = report_data['weighted avg']['f1-score']
    precision = report_data['weighted avg']['precision']
    print(f"AKurasi Model : {accuracy * 100:.2f}%")
    metrics = {'accuracy' : accuracy, 'f1_score' : report_data, 'f1_weighted':f1_weighted, 'precision' : precision }
    joblib.dump(metrics, 'models/metrics_dt.pkl')
    print("SUKSES! Metris tersimpan di 'models/metrics_dt.pkl'")


# visualisasi
def visualisasi(model, feature_names):
    if model is None:
        print("Model belum di buat, jalankan training dulu.")
    
    class_names = [str(c) for c in model.classes_]
    fig = plt.figure(figsize=(25, 20), dpi=300)
    pict_tree = tree.plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, fontsize=10, rounded=True)
    outputh_path_png = 'reports/decision_tree.png'
    outputh_path_pdf = 'reports/decision_tree.pdf'
    fig.savefig(outputh_path_png, bbox_inches='tight')
    fig.savefig(outputh_path_pdf, bbox_inches='tight')
    print(f"Gambar decision tree berhasil di simpan di : {outputh_path_png} & {outputh_path_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    model, feature_names, x_test, y_test = train()
    akurasi(model, x_test, y_test)
    visualisasi(model, feature_names)




