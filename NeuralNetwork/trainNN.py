import tensorflow as tf
import tensorflow.compat.v1 as tfc
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

tfc.disable_eager_execution()

# get data from training and testing datasets
def load_dataset():
  # Load training and testing datasets
  training = pd.read_csv('training_data.csv')
  testing = pd.read_csv('test_data.csv')

  Xtrain_maybe = training.iloc[1:,4:523].to_numpy().astype(np.float)
  Ytrain_maybe = training.iloc[1:, 523:524].to_numpy().astype(np.float)
  
  XValid_maybe = testing.iloc[1:,4:523].to_numpy().astype(np.float)
  YValid_maybe = testing.iloc[1:, 523:524].to_numpy().astype(np.float)
  print("Xtrain", Xtrain_maybe.shape)

  # get rid of maybe values
  Xtrain_ = []
  Ytrain_ = []
  for i in range(0, len(Xtrain_maybe)):
    if Ytrain_maybe[i] != 0:
      Xtrain_.append(Xtrain_maybe[i])
      Ytrain_.append((Ytrain_maybe[i] + 1)/2)

  # change values from -1 and 1 to 0 and 1
  XValid = []
  YValid = []
  for i in range(0, len(XValid_maybe)):
    if YValid_maybe[i] != 0:
      XValid.append(XValid_maybe[i])
      YValid.append((YValid_maybe[i] + 1)/2)

  # print("Xtrain_", len(Xtrain_))
  # print("Ytrain_", len(Ytrain_))
  # print(Ytrain_)

  # change mix of correct to incorrect
  Xtrain = []
  Ytrain = []
  amount_correct = 3500
  amount_incorrect = 3500
  current_correct = 0
  current_incorrect = 0
  for i in range(0, len(Xtrain_)):
    if(Ytrain_[i][0] == 1 and current_correct < amount_correct):
      Ytrain.append(Ytrain_[i])
      Xtrain.append(Xtrain_[i])
      current_correct += 1
    elif(Ytrain_[i][0] == 0 and current_incorrect < amount_incorrect):
      Ytrain.append(Ytrain_[i])
      Xtrain.append(Xtrain_[i])
      current_incorrect += 1


  # chage to numpy arrays
  Xtrain = np.array(Xtrain)
  Ytrain = np.array(Ytrain)
  XValid = np.array(XValid)
  YValid = np.array(YValid)


  # print metrics and data
  print("Xtrain", Xtrain.shape)
  print(Xtrain)
  print("XValid", XValid.shape)
  print(XValid)
  print("Ytrain", Ytrain.shape)
  print(Ytrain)
  print("YValid", YValid.shape)
  print(YValid)

  return Xtrain, XValid, Ytrain, YValid


X_, X_t, Y_, Y_t = load_dataset()

# create the TF neural net
# some hyperparams
training_epochs = 200

n_neurons_in_h1 = 500
n_neurons_in_h2 = 250
n_neurons_in_h3 = 100
n_neurons_in_h4 = 50
n_neurons_in_h5 = 10
learning_rate = 0.0005

n_features = len(X_[0])
labels_dim = 1
#############################################

# these placeholders serve as our input tensors
x = tfc.placeholder(tf.float32, [None, n_features], name='input')
y = tfc.placeholder(tf.float32, [None, labels_dim], name='labels')

#weights and biases for layer 1
W1 = tf.Variable(tfc.truncated_normal([n_features, n_neurons_in_h1],
 mean=0, stddev=1 / np.sqrt(n_features)),name='weights1')
b1 = tf.Variable(tfc.truncated_normal([n_neurons_in_h1], 
  mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
#out put of layer 1
y1 = tf.nn.leaky_relu((tf.matmul(x, W1) + b1),
 n name='activationLayer1')

#layer 2
W2 = tf.Variable(tfc.random_normal([n_neurons_in_h1, n_neurons_in_h2],
 mean=0, stddev=1),name='weights2')
b2 = tf.Variable(tfc.random_normal([n_neurons_in_h2],
 mean=0, stddev=1), name='biases2')
y2 = tf.nn.leaky_relu((tf.matmul(y1, W2) + b2),
 name='activationLayer2')

#layer 3
W3 = tf.Variable(tfc.random_normal([n_neurons_in_h2, n_neurons_in_h3],
 mean=0, stddev=1),name='weights3')
b3 = tf.Variable(tfc.random_normal([n_neurons_in_h3],
 mean=0, stddev=1), name='biases3')
y3 = tf.nn.leaky_relu((tf.matmul(y2, W3) + b3),
 name='activationLayer3')

#layer 4
W4 = tf.Variable(tfc.random_normal([n_neurons_in_h3, n_neurons_in_h4],
 mean=0, stddev=1),name='weights4')
b4 = tf.Variable(tfc.random_normal([n_neurons_in_h4],
 mean=0, stddev=1), name='biases4')
y4 = tf.nn.leaky_relu((tf.matmul(y3, W4) + b4),
 name='activationLayer4')

#layer 5
W5 = tf.Variable(tfc.random_normal([n_neurons_in_h4, n_neurons_in_h5],
 mean=0, stddev=1),name='weights5')
b5 = tf.Variable(tfc.random_normal([n_neurons_in_h5],
 mean=0, stddev=1), name='biases5')
y5 = tf.nn.leaky_relu((tf.matmul(y4, W5) + b5),
 name='activationLayer5')

# output layer weights and biases
Wo = tf.Variable(tfc.random_normal([n_neurons_in_h5, labels_dim],
 mean=0, stddev=1 ),name='weightsOut')
bo = tf.Variable(tfc.random_normal([labels_dim],
 mean=0, stddev=1), name='biasesOut')

# logits and loss fuction
logits = (tf.matmul(y5, Wo) + bo)
loss = tf.reduce_mean(tfc.nn.sigmoid_cross_entropy_with_logits(
  labels = y, logits = logits))

# tap a separate output that applies 
# softmax activation to the output layer
# for training accuracy readout
a = tf.nn.sigmoid(logits, name='activationOutputLayer')

# optimizer used to compute gradient of 
# loss and apply the parameter updates.
# the train_step object returned is ran by a 
# TF Session to train the net
train_step = tfc.train.AdamOptimizer(learning_rate).minimize(loss)

# prediction accuracy
# compare predicted value from network with the expected value/target
correct_prediction = tf.equal(tf.round(a), y)
# accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
 name="Accuracy")

#############################################
# ***NOTE global_variables_initializer() must be called 
# before creating a tf.Session()!***
training_loss = []
testing_loss = []
training_accuracy = []
testing_accuracy = []
init_op = tfc.global_variables_initializer()
saver = tfc.train.Saver(var_list = None)
# create a session for training and feedforward (prediction).
# Sessions are TF's way to run
# feed data to placeholders and variables, 
# obtain outputs and update neural net parameters
with tfc.Session() as sess:
    #initialization of all variables
    sess.run(init_op)

    # create number for batches
    batch_size = 128
    batches = int(len(X_) / batch_size)
    # training loop over the number of epochs
    for epoch in range(training_epochs):
        losses = 0
        accs = 0
        # loop through batches
        for j in range(batches):
            idx = np.random.randint(X_.shape[0], size=batch_size)
            X_b = X_[idx]
            Y_b = Y_[idx]

            # train the network, 
            # note the dictionary of inputs and labels
            sess.run(train_step, feed_dict={x: X_b, y: Y_b})
            # feedforwad the same data and labels, but grab the accuracy and loss as outputs
            acc, l, soft_max_a = sess.run([accuracy, loss, a],
             feed_dict={x: X_b, y: Y_b})
            # print(np.sum(l))
            losses = losses + np.sum(l)
            accs = accs + np.sum(acc)
        print("Epoch %.8d " % epoch, "avg train loss over", batches,
         " batches ", "%.4f" % (losses/batches),"avg train acc ",
         "%.4f" % (accs/batches*100))
        training_loss.append(losses/batches)
        training_accuracy.append(accs/batches * 100)
        # test on the holdout set
        acc, l, soft_max_a = sess.run([accuracy, loss, a],
         feed_dict={x: X_t, y: Y_t})
        print("Epoch %.8d " % epoch, "test loss %.4f" % np.sum(l),
          "test acc %.4f" % (acc*100))
        testing_loss.append(np.sum(l))
        testing_accuracy.append(acc * 100)
    # saver.save(sess, 'trainedNN')    

#print output values
Y_t = Y_t.flatten()
soft_max_a = soft_max_a.flatten()
print(Y_t)

y_pred = []
for pred in soft_max_a:
  if pred < .5:
    y_pred.append(0)
  else:
    y_pred.append(1)

y_act = []
for act in Y_t:
  if act == 0:
    y_act.append(0)
  else:
    y_act.append(1)

# create confustion matrix and classification report
conf = confusion_matrix(y_act,y_pred)
rep = classification_report(y_act, y_pred, labels = [0, 1])
print(rep)

# graph accuracy plots
plt.figure(1)
iterations = list(range(training_epochs))
plt.plot(iterations, training_accuracy, label='training')
plt.plot(iterations, testing_accuracy, label='testing')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.legend()

#graph loss plots
plt.figure(2)
iterations = list(range(training_epochs))
plt.plot(iterations, training_loss, label='training')
plt.plot(iterations, testing_loss, label='testing')
plt.ylabel('Loss')
plt.xlabel('iterations')
plt.legend()

#graph confusion matrix
plt.figure(3)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in conf.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in 
  conf.flatten()/np.sum(conf)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,
  group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Y Predicted')
plt.ylabel('Y Actual')

plt.show()