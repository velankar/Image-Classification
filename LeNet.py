import tensorflow as tf
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix
CIFAR_DIR = 'cifar-100-python/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

all_data = unpickle(CIFAR_DIR+"train")
X = all_data[b"data"]
X_train =X[:40000]
X_validation = X[40000:50000]

X_train = X_train.reshape(40000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")/255
X_validation = X_validation.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")/255
X_mean = np.mean(X_train,axis=0,dtype= np.float32)
X_train = X_train - X_mean
X_validation = X_validation -X_mean
Y = all_data[b"fine_labels"]

def one_hot_encode(vec, vals=100):
    '''
    For use to one-hot encode the 100- possible labels
    '''
    n = len(vec)
    #print(n)
    out = np.zeros((n, vals))
    
    out[range(n), vec] = 1
    return out

output = one_hot_encode(Y,100)
Y_train = output[:40000]
Y_validation = output[40000:50000]

all_test = unpickle(CIFAR_DIR+"test")
test_X = all_test[b"data"]
test_Y = all_test[b"fine_labels"]
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")/255
test_X = test_X - X_mean
test_Y = one_hot_encode(test_Y,100)
test_superclass = all_test[b"coarse_labels"]
epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]

def next_batch(batch_size):

    global X_train
    global Y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], Y_train[start:end]

x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,100])

def init_weights(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
def init_bias(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
def conv2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')
def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def convolution_layer(input_x, shape):
    W =init_weights(shape)
    b =init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)
def full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    weights = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, weights) + b

def rank5(preds,correct):
    preds = np.argsort(preds,axis=1)
    #print(preds)
    rank_5_acc=0
    sorted_pred = np.flip(preds,axis=1)
    #sorted_pred[:,:5]
    for i in range(len(correct)):
        if correct[i] in sorted_pred[i,:5]:
            rank_5_acc+=1
    aaacc = float(rank_5_acc)/len(correct)
    print("Rank 5 Accuracy for fine labels is: ",aaacc)
    

def plotfig(step_t,loss_v,loss_t,filename_training,filename_validation):

    plt.plot(step_t, loss_t, 'r--')
    plt.plot(step_t,loss_v,'b--')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Loss")
    plt.savefig('plot.png')
mapping={0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],
        4:[0,51,53,57,83],5:[22,39,40,86,87], 6:[5,20,25,84,94],7:[6,7,14,18,24],
        8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],
        12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],
        16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}

def superclass_accuracy(mapping,ypred,ycorrect):
    super_y =[]
    for i in range(len(ypred)):
        for key,values in mapping.items():
            if ypred[i] in values:
                super_y.append(key)
    #print(ypred)
    confusion_mat = confusion_matrix(y_true=ycorrect,y_pred=super_y)
    print(confusion_mat)
        # CLASSWISE ACCURACY
    col_sum = confusion_mat.sum(axis=1)
    classwise_acc = (confusion_mat/col_sum).diagonal()
    print("Classwise accuracy",classwise_acc)
    print("rank1 accuracy for superclass is ",np.trace(confusion_mat)/10000)


def superclassrank5(mapping,ypredlist,ycorrect):
    rank_5_acc=0
    ycorr =[]
    for i in range(len(ycorrect)):
        for key,values in mapping.items():
            if ycorrect[i] in values:
                ycorr.append(key)
    ypredlist = np.argsort(ypredlist,axis=1)
    #print(preds)
    y_super= np.zeros((10000,5))
    sorted_pred = np.flip(ypredlist,axis=1)
    sorted_pred = sorted_pred[:,:5]
    for i in range(len(ycorrect)):
        for j in range(5):
            for key,values in mapping.items():
                if sorted_pred[i][j] in values:
                    y_super[i][j] = key
        if ycorr[i] in y_super[i,:5]:
            rank_5_acc+=1
    aacc = float(rank_5_acc)/len(ycorrect)
    print("rank 5 accuracy for super class is ",aacc)
def incorrect_matches(ypred,ycorrect):
    corr=[]
    wrong=[]
    for i in range(len(ycorrect)):
        if ypred[i] == ycorrect[i]:
            corr.append(i)
        else:
            wrong.append(i)
        if len(corr)>=5 and len(wrong)>=5:
            return corr,wrong
    return

convo_1 = convolution_layer(x,[5,5,3,6])
convo_1_pooling = maxpool(convo_1)
convo_2 = convolution_layer(convo_1_pooling,[5,5,6,16])
convo_2_pooling =maxpool(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*16])
full_layer_one = tf.nn.relu(full_layer(convo_2_flat,120))
full_layer_two = tf.nn.relu(full_layer(full_layer_one,84))
y_pred = full_layer(full_layer_two,100)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

y_correct = tf.argmax(y_true,1)
y_predicted = tf.argmax(y_pred,1)
loss_list_train=[]
loss_list_validation=[]
epochs =[]
error_train =0.0
error_validation=0.0
with tf.Session() as sess:
    i=0
    j=0
    global epochs_completed
    sess.run(tf.global_variables_initializer())
    while i<6251:
        batch = next_batch(64)
        
        sess.run(train,feed_dict={x: batch[0], y_true: batch[1]})
        error_train += sess.run(cross_entropy,feed_dict={x:batch[0],y_true:batch[1]})
        error_validation += sess.run(cross_entropy,feed_dict = {x:X_validation,y_true:Y_validation})
        if i%312 == 0:
            
            print('Currently on epoch',epochs_completed)
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            #error_train = sess.run(cross_entropy,feed_dict={x:batch[0],y_true:batch[1]})
            #print("Error_train",error_train)
            #error_validation = sess.run(cross_entropy,feed_dict = {x:X_validation,y_true:Y_validation})
            print("Error_validation:",error_validation/312)
           # acc1, cross_entropy1 = sess.run([acc,cross_entropy],feed_dict={x:X_train,y_true:Y_train})
            loss_list_train.append(error_train/312)
            #print(loss_list_validation)
            print("Accuracy on validation data")
            print(sess.run(acc,feed_dict={x:X_validation,y_true:Y_validation}))
            loss_list_validation.append(error_validation/312)
            error_train =0.0
            error_validation =0.0
            epochs.append(j)
            j+=0.5
            print('\n')
        i+=1
    
    print(sess.run(acc,feed_dict={x:test_X,y_true:test_Y}))
    #y_correct = sess.run(y_correct,feed_dict={x:test_X,y_true:test_Y})
    y_correct = all_test[b"fine_labels"]
    y_predicted = sess.run(y_predicted,feed_dict={x:test_X,y_true:test_Y})
    y_predlist = sess.run(y_pred,feed_dict={x:test_X,y_true:test_Y}) 
    #print(y_predicted)
    #CONFUSION MATRIX
    confusion_mat = confusion_matrix(y_true=y_correct,y_pred=y_predicted)
    print("Rank 1 accuarcy for fine labels is ",np.trace(confusion_mat)/10000)
    print(confusion_mat)
    # CLASSWISE ACCURACY
    col_sum = confusion_mat.sum(axis=1)
    classwise_acc = (confusion_mat/col_sum).diagonal()
    print("class_wise accuaracy",classwise_acc)
    rank5(y_predlist,y_correct)
    #print(epochs)
    #print(loss_list_validation)
    #print(len(epochs),len(loss_list_validation))
    superclass_accuracy(mapping,y_predicted,test_superclass)
    superclassrank5(mapping,y_predlist,y_correct)
    filename_training = "trainingloss.png"
    filename_validation = "validationloss.png"
    plotfig(epochs[1:],loss_list_validation[1:],loss_list_train[1:],filename_training,filename_validation)
    c,w =  incorrect_matches(y_predicted,y_correct)
    print("correrct",c)
    print("wrong",w)
    
