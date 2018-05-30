import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#Data process
test_x = pd.read_csv("Processedtest.csv")
test_Passenger_ID = test_x["PassengerId"]
test_x = test_x.drop(["PassengerId"], axis=1)
Passenger_id = test_Passenger_ID.copy()
evaluation = Passenger_id.to_frame()
data_x = pd.read_csv("Processedtrain.csv")
data_x = data_x.drop(["PassengerId"], axis=1)
data_y = data_x["Survived"]

data_y2 = []
for i in range(len(data_y)):
    if data_y[i] ==1:
        data_y2.append([0.0,1.0])
    else:
        data_y2.append([1.0,0.0])

data_x = data_x.drop(["Survived"], axis=1)
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y2, test_size=0.2)

#Logistic regression model

epochs = 100
L_rate = 0.003
batch_size = 50
x = tf.placeholder(tf.float32, [None,train_x.shape[1]],name ="Input_x")
y_ = tf.placeholder(tf.float32, [None,2],name = "label")

W = tf.Variable(tf.truncated_normal([train_x.shape[1],2], stddev=0.1), name="Weight")
b = tf.Variable(tf.truncated_normal([2], stddev=0.1), name="bias")
y = tf.nn.softmax(tf.matmul(x, W) + b, name="predict_label")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_algorithm = tf.train.GradientDescentOptimizer(learning_rate = L_rate).minimize(cross_entropy)
predict = tf.argmax(y,1)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    best_result = 0
    while epoch <= epochs:
        print("epoch of training", epoch)
        for i in range(0, 700, batch_size):
            sess.run(train_algorithm, feed_dict={x: train_x[i:i+batch_size], y_: train_y[i:i+batch_size]})
            loss = sess.run(cross_entropy, feed_dict={x: train_x[i:i+batch_size], y_: train_y[i:i+batch_size]})

            #predict accuracy
            same_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(same_prediction, tf.float32))

            accuracy_train = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            accuracy_valid = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            print("==================================================")
            print("Batch", i, "~",i + batch_size, "Of", epoch, "epochs")
            print("Accuracy on train set" , accuracy_train )
            print("Accuracy on validate set" , accuracy_valid ,", Loss = ", loss)
            print("Current best accuracy" , best_result)

            if best_result < sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y}):
                best_result = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
                predict_result = sess.run(predict, feed_dict={x: test_x})

        epoch += 1

    print ("The Final best accuracy: ", best_result)
    evaluation["Survived"] = predict_result
    evaluation.to_csv("submission.csv", index = False)