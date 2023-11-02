import numpy as np
import requests, gzip, os, hashlib

#fetch data
path='D:/Users/Dose/Documents/AI/data'
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

#Train set from MNIST - random split for Train and Validation sets
X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]

rand = np.arange(60000)
np.random.shuffle(rand)

train_no = rand[:50000]
val_no = np.setdiff1d(rand, train_no)

X_train, X_val = X[train_no, :, :], X[val_no, :, :]
Y_train, Y_val = Y[train_no], Y[val_no]

#Test set from MNIST
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

#Neural network
#Input Layer - 784 units - every pixel of input image 28x28 size
#Hidden Layer - 128 units
#Output Layetr - 10 units - predicted digit

def init_layer(x, y):
    layer = np.random.uniform(-1., 1., size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)

#np.random.seed(42)
l1 = init_layer(28*28, 128)
l2 = init_layer(128, 10)

#Sigmoid Function - used to change values from input layer to values between 0 and 1
def sigmoid(x):
    return 1/(np.exp(-x)+1)

def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax Function - create probabilities, normalize results of output vector
def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)

def d_softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)*(1-exps/np.sum(exps))

l2_sample_output = np.array([12, 34, -67, 23, 0, 134, 76, 24, 78, -98])
l2_normalized = softmax(l2_sample_output)
max_l2_val = np.argmax(l2_normalized)
print(max_l2_val, l2_sample_output[max_l2_val])

#Forward/Backward Pass algorithm
def forward_backward_pass(x, y):
    pass