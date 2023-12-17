from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from PIL import Image

app=Flask(__name__)

model=load_model('best_Final.h5')
model.make_predict_function()   
#dic={0:'Arborio', 1:'Basmati', 2:'Ipsala', 3:'Jasmine', 4:'Karacadag'}
label = ['Arborio Bad Quality','Arborio Good Quality','Basmati Bad Quality','Basmati Good Quality', 'Ipsala Bad Quality', 'Ipsala Good Quality','Jasmine Good Quality','Jasmine Bad Quality' 'Karacadag Good Quality','Karacadag Bad Quality']
le = preprocessing.LabelEncoder()
le.fit(label)




def predict_label(img_path):
    # Load the image with target size of 400x400
    #i = image.load_img(img_path, target_size=(400, 400))

    # Convert the image to a numpy array and normalize it
    #i = image.img_to_array(i) / 255.0

    i = tf.io.read_file(img_path)
    i = tf.image.decode_jpeg(i, channels=3)
    i = i/255
    i = tf.image.convert_image_dtype(i, tf.float32)
    i = tf.image.resize(i, [400, 400])

    # Reshape the array to add a fourth dimension (for batch size)
    #i = i.reshape(None, 400, 400, 3)] 

    # Predict the class of the image
    p = list(le.classes_)[np.argmax(model.predict(i[np.newaxis,:,:,:],verbose=0))]

    # Get the index of the highest probability
    #result = list(le.classes_)p

    # Return the corresponding label from your dictionary
    return p



@app.route('/')
def index():
    return render_template('home.html')


@app.route("/check", methods=["POST"])
def check():
    if request.method=='POST':
        f = request.files['my_image']
        img_path=f.filename
        f.save(img_path)
        p = predict_label(img_path)
    return render_template('demo.html', prediction=p, img_path=img_path)
      
if __name__=="__main__":
    app.run(debug=True)
