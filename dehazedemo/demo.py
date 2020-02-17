from flask import *  
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.losses import MSE
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)  
app.config["IMAGE_UPLOADS"] = "D:/yeanwei97/Downloads/dehazedemo/static"
app.config["MODEL"] = "D:/yeanwei97/Downloads/dehazedemo/model"

def lossVGG(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = 'block2_conv2'

    lossModel = Model(vgg.input, vgg.get_layer(content_layers).output)

    vggX = lossModel(y_pred)
    vggY = lossModel(y_true)
    
    return K.mean(K.square(vggX - vggY)) 

def my_loss(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    return lossVGG(y_true, y_pred) + mse

def resize(img):
  x, y = img.shape[0], img.shape[1]

  while (x % 16 != 0):
    x = x-1

  while (y % 16 != 0):
    y = y-1

  img = cv2.resize(img, (y, x))

  return img

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

global o, i, c, graph, sess
config = tf.ConfigProto()
sess = tf.Session(config=config)
graph = tf.get_default_graph()

set_session(sess)
o = load_model(os.path.join(app.config["MODEL"], 'outdoor.hdf5'), custom_objects={"my_loss": my_loss})
i = load_model(os.path.join(app.config["MODEL"], 'indoor.hdf5'), custom_objects={"my_loss": my_loss})
c = load_model(os.path.join(app.config["MODEL"], 'combine.hdf5'), custom_objects={"my_loss": my_loss})

@app.route('/')  
@app.route('/home')  
def home():  
    return render_template('dehaze.html', i = None, o = None)

@app.route('/dehaze', methods=['GET', 'POST'])
def dehaze():
    if request.method == 'POST':
        f = request.files['file']  
        name = os.path.join(app.config["IMAGE_UPLOADS"], f.filename)
        f.save(name)

        x = cv2.imread(name)
        test = resize(x)
        I = test/255.
        test_input = np.expand_dims(I, axis=0)

        with graph.as_default():
            set_session(sess)
            outdoor = o.predict(test_input)
            indoor = i.predict(test_input)
            combine = c.predict(test_input)

        J1 = cv2.resize(outdoor[0], (x.shape[1], x.shape[0]))*255
        J2 = cv2.resize(indoor[0], (x.shape[1], x.shape[0]))*255
        J3 = cv2.resize(combine[0], (x.shape[1], x.shape[0]))*255

        n1 = f.filename.split('.')[0] + '_outdoor.png'
        n2 = f.filename.split('.')[0] + '_indoor.png'
        n3 = f.filename.split('.')[0] + '_combine.png'

        out_name1 = os.path.join(app.config["IMAGE_UPLOADS"], n1)
        out_name2 = os.path.join(app.config["IMAGE_UPLOADS"], n2)
        out_name3 = os.path.join(app.config["IMAGE_UPLOADS"], n3)

        out_arr = [n1, n2, n3]

        cv2.imwrite(out_name1, J1)
        cv2.imwrite(out_name2, J2)
        cv2.imwrite(out_name3, J3)

        return render_template("dehaze.html", i = f.filename, o = out_arr) 

    return render_template('dehaze.html', i = None, o = None)

  
if __name__ == '__main__':  
    app.run(debug = True)