import logging
import shutil
import time
import datetime
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
from cassandra.cluster import Cluster
app = Flask(__name__)


log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

KEYSPACE = "mnist"
cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
session = cluster.connect()

log.info("Creating keyspace: mnist...")
try:
    session.execute("""
        CREATE KEYSPACE %s
        WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
        """ % KEYSPACE)

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    log.info("creating table...")
    session.execute("""
        CREATE TABLE uploadedimages (
            upload_time text,
            image_name text,
            prediction text,
            PRIMARY KEY (upload_time, image_name)
        )
        """)
except Exception as e:
    log.error("Unable to create keyspace")
    log.error(e)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        fname = secure_filename(f.filename)
        f.save(fname)
        imvalue = imageprepare(fname)#预测图片数字
        predint = str(predictint(imvalue)[0])
        prediction_show = 'the number is ' + predint
        ext = fname.rsplit('.', 1)[1]  # 获取文件后缀

        unix_time = int(time.time())
        new_filename = str(unix_time) + '.' + ext  # 更改上传的文件名
        shutil.move(f.filename, new_filename)
        shutil.move(new_filename, "uploaded_images/")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        #向数据库传数据
        session.execute("""
                USE %s
                """ % KEYSPACE)
        session.execute(
            """
            INSERT INTO uploadedimages (upload_time, image_name, prediction)
            VALUES (%s, %s, %s)
            """,
            (current_time, new_filename, predint)
        )

        return prediction_show + ' and your file is uploaded successfully'
    else:
        return render_template('upload.html')


def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """

    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    """
    Load the model2.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.

    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        # print ("Model restored.")

        prediction = tf.argmax(y_conv, 1)
        return prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)


def imageprepare(fname):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(fname).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
    # print(tva)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = True, use_reloader = False)
