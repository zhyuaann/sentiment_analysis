from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('sentiment_pipeline.pkl')

label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    comment = ""

    if request.method == 'POST':
        comment = request.form['comment']
        pred = model.predict([comment])[0]
        prediction = label_map.get(pred, 'Tidak Diketahui')

    return render_template('index.html', prediction=prediction, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
