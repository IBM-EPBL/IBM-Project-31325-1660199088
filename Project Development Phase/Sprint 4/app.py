from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

from cloudant.client import Cloudant

client = Cloudant.iam('947534e7-5837-41fa-b216-7aaace1a2275-bluemix', '6b6UgDIZZDVt0BklyqhcQzjVEQZIpws6YOGgzjb2Tg8U',
                      connect=True)
my_database = client.create_database('my_database')

model = load_model(r"Updated-xception-diabetic-retinopathy.h5")


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/register.html", methods=['GET', 'POST'])
@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        x = [x for x in request.form.values()]
        # print(x)
        data = {
            '_id': x[1],  # setting id is optional
            'name': x[0],
            'paw': x[2]
        }
        # print(data)

        query = {'_id': {'Seq': data['_id']}}
        docs = my_database.get_query_result(query)
        # print(docs)

        if len(docs.all()) == 0:
            print(my_database.create_document(data))
            return render_template('prediction.html', pred="Registration Successful,please login using your details")
        else:
            return render_template('login.html', pred="You are already a member, please login using your details")
    else:
        return render_template("register.html")


# Login page
@app.route('/login', methods=['GET', 'POST'])
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['_id']
        passw = request.form['psw']
        print(user, passw)

        query = {'_id': {'$eq': user}}

        docs = my_database.get_query_result(query)
        print(docs.all())

        if len(docs.all()) == 0:
            return render_template('login.html', pred="The username is not found.")
        else:
            if user == docs[0][0]['_id'] and passw == docs[0][0]['paw']:
                return redirect(url_for('prediction'))
            else:
                print('Invalid User')

    else:
        return render_template('login.html')


# Logout
@app.route('/logout')
@app.route('/logout.html')
def logout():
    return render_template('logout.html')


# Prediction page
@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == 'POST':
        print(request.files.keys())
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads',
                                f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['No Diabetic Retinopathy, Just Relax', 'Mild DR, Time for a basic checkup', 'Moderate DR, Condult a doctor', 'Severe DR,Check with your doctor immediately', 'Proliferative DR,Check with your doctor immediately']
        result = str(index[prediction[0]])
        return render_template('prediction.html', prediction=result)
    else:
        return render_template('prediction.html')


#


if __name__ == "__main__":
    app.run(port=5000, debug=True)