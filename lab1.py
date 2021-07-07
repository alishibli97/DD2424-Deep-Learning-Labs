import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
functions
"""


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Datasets/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no = W.shape[0]
	d = X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = ComputeCost(X, Y, W, lamda, b)

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, lamda, b_try)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i, j] += h
			c2 = ComputeCost(X, Y, W_try, lamda, b)
			grad_W[i, j] = (c2-c) / h

	return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no = W.shape[0]
	d = X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, lamda, b_try)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, lamda, b_try)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i, j] -= h
			c1 = ComputeCost(X, Y, W_try, lamda, b)

			W_try = np.array(W)
			W_try[i, j] += h
			c2 = ComputeCost(X, Y, W_try, lamda, b)

			grad_W[i, j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]


def montage(W, name):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2, 5)
	for i in range(2):
		for j in range(5):
			im = W[i*5+j, :].reshape(32, 32, 3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1, 0, 2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.savefig(name)  # plt.show()


num_labels = 10


## 1.0
def _LoadBatch(file):
    data = LoadBatch(file)
    X = data[b'data'].T/255
    y = data[b'labels']

    # one hot encoding
    Y = np.zeros(shape=(num_labels, len(y)))
    for i, label in enumerate(y):
        Y[label, i] = 1

    return X, Y, y

## 2.0
def normalize(train_X, val_X, test_X):
    std = train_X.std(axis=1).reshape(train_X.shape[0], 1)
    mean = train_X.mean(axis=1).reshape(train_X.shape[0], 1)
    
    train_X = (train_X-mean)/std
    val_X = (val_X-mean)/std
    test_X = (test_X-mean)/std

    return train_X, val_X, test_X

## 3.0
def initialize_weights():
    np.random.seed(400)
    W = np.random.normal(
        size=(num_labels, train_X.shape[0]), loc=0, scale=0.01)
    b = np.random.normal(size=(num_labels, 1), loc=0, scale=0.01)
    return W, b

## 4.0
def EvaluateClassifier(X, W, b=0, loss='cross_entropy'):
    if loss == 'cross_entropy': return softmax(np.dot(W, X)+b)
    elif loss == 'svm': return np.dot(W, X)

## 5.0
def ComputeCost(X, Y, W, _lambda, b=0, loss='cross_entropy'):
    if loss == 'cross_entropy':
        P = EvaluateClassifier(X, W, b, loss)
        l_cross = -sum(np.log((Y*P).sum(axis=0)))
        l_regularization = (W**2).sum()
        J = (1/X.shape[1])*l_cross + _lambda*l_regularization
    elif loss == 'svm':
        S = EvaluateClassifier(X, W, loss=loss)
        margins = np.maximum(0, (S-np.sum(S*Y, axis=0)+1)*(1-Y))
        l_svm = np.sum(margins)
        l_regularization = (W**2).sum()
        J = (1/X.shape[1])*l_svm + _lambda*l_regularization
    return J

## 6.0
def ComputeAccuracy(X, y, W, b=1, loss='cross_entropy'):
    if loss == 'cross_entropy': P = EvaluateClassifier(X, W, b)
    elif loss == 'svm': P = EvaluateClassifier(X, W, loss=loss)
    acc = np.mean(y == np.argmax(P, 0))
    return acc

## 7.0
def ComputeGradients(X, Y, y, P, W, _lambda, loss='cross_entropy'):
    n = X.shape[1]
    if loss == 'cross_entropy':
        G = -(Y-P)
        # dJ/dw = (1/n) G.X^T + 2*lambda*W
        grad_W = (1/n)*(np.dot(G, X.T))+2*_lambda*W
        # dJ/db = (1/n)*G*1
        grad_b = (1/n)*(np.dot(G, np.ones(shape=(n, 1)))).reshape(num_labels, 1)
        return grad_W, grad_b
    elif loss == 'svm':
        margins = np.maximum(0, (P-np.sum(P*Y, axis=0)+1)*(1-Y))
        dL_dS = margins
        dL_dS[margins > 0] = 1
        dL_dS[y, np.arange(n)] = -np.sum(dL_dS, axis=0)
        grad_W = (1/n)*np.dot(dL_dS,X.T)
        grad_W += 2*_lambda*W
        return grad_W

def testing_gradients(X, Y, y, P, W, b, _lambda, h=1e-6):

    grad_w_analytical, grad_b_analytical = ComputeGradients(X, Y, y, P, W, _lambda)
    grad_w_num, grad_b_num = ComputeGradsNum(X, Y, P, W, b, _lambda, h)
    grad_w_num_slow, grad_b_num_slow = ComputeGradsNumSlow(X, Y, P, W, b, _lambda, h)

    epsilon = 1e-6

    diff_w_computed_num = np.abs(grad_w_num-grad_w_analytical)
    diff_b_computed_num = np.abs(grad_b_num-grad_b_analytical)
    diff_w_computed_num_slow = np.abs(grad_w_num_slow-grad_w_analytical)
    diff_b_computed_num_slow = np.abs(grad_b_num_slow-grad_b_analytical)

    error_w_num = np.sum(diff_w_computed_num > 1e-6)
    error_w_num_acc = np.mean(diff_w_computed_num > 1e-6)
    error_b_num = np.sum(diff_b_computed_num > 1e-6)
    error_b_num_acc = np.mean(diff_b_computed_num > 1e-6)
    error_w_num_slow = np.sum(diff_w_computed_num_slow > 1e-6)
    error_w_num_slow_acc = np.mean(diff_w_computed_num_slow > 1e-6)
    error_b_num_slow = np.sum(diff_b_computed_num_slow > 1e-6)
    error_b_num_slow_acc = np.mean(diff_b_computed_num_slow > 1e-6)

    error_rel_1 = diff_w_computed_num / np.maximum(epsilon, np.abs(grad_w_num)+np.abs(grad_w_analytical))
    error_rel_2 = diff_b_computed_num / np.maximum(epsilon, np.abs(grad_b_num)+np.abs(grad_b_analytical))
    error_rel_3 = diff_w_computed_num_slow / np.maximum(epsilon, np.abs(grad_w_num_slow)+np.abs(grad_w_analytical))
    error_rel_4 = diff_b_computed_num_slow / np.maximum(epsilon, np.abs(grad_b_num_slow)+np.abs(grad_b_analytical))

    error_relative_w_num = np.sum(error_rel_1 > epsilon)
    error_relative_w_num_acc = np.mean(error_rel_1 > epsilon)
    error_relative_b_num = np.sum(error_rel_2 > epsilon)
    error_relative_b_num_acc = np.mean(error_rel_2 > epsilon)
    error_relative_w_num_slow = np.sum(error_rel_3 > epsilon)
    error_relative_w_num_slow_acc = np.mean(error_rel_3 > epsilon)
    error_relative_b_num_slow = np.sum(error_rel_4 > epsilon)
    error_relative_b_num_slow_acc = np.mean(error_rel_4 > epsilon)

    print("Using error as number of points where difference is less than epsilon = 1e-6", end='\n\n')

    print("Between ComputeGradients and ComputeGradNum:")
    print(f"There are {error_w_num} errors for weights (error rate = {error_w_num_acc*100}%)")
    print(f"The are {error_b_num} errors for biases (error rate = {error_b_num_acc*100}%)")

    print()

    print("Between ComputeGradients and ComputeGradNumSlow:")
    print(f"There are {error_w_num_slow} errors for weights (error rate = {error_w_num_slow_acc*100}%)")
    print(f"The are {error_b_num_slow} errors for biases (error rate = {error_b_num_slow_acc*100}%)")

    print()

    print("Using relative error as number of points where difference is less than epsilon = 1e-6", end='\n\n')

    print("Between ComputeGradients and ComputeGradNum:")
    print(f"There are {error_relative_w_num} errors for weights (error rate = {error_relative_w_num_acc*100}%)")
    print(f"The are {error_relative_b_num} errors for biases (error rate = {error_relative_b_num_acc*100}%)")

    print()

    print("Between ComputeGradients and ComputeGradNumSlow:")
    print(f"There are {error_relative_w_num_slow} errors for weights (error rate = {error_relative_w_num_slow_acc})")
    print(f"The are {error_relative_b_num_slow} errors for biases (error rate = {error_relative_b_num_slow_acc})")


## 8.0
def MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, early_stopping=False, weight_decay=False, loss='cross_entropy'):
    n = train_X.shape[1]
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']

    loss_threshold = 0.4

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(n_epochs):
        for j in range(n//n_batch):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            inds = range(j_start, j_end)

            X_batch = train_X[:, inds]
            Y_batch = train_Y[:, inds]
            y_batch = [train_y[index] for index in inds]

            P_batch = EvaluateClassifier(X_batch, W, b, loss)
            if loss == 'cross_entropy':
                grad_W, grad_b = ComputeGradients(
                    X_batch, Y_batch, y_batch, P_batch, W, _lambda, loss)
                W += -eta*grad_W
                b += -eta*grad_b
            elif loss == 'svm':
                grad_W = ComputeGradients(
                    X_batch, Y_batch, y_batch, P_batch, W, _lambda, loss)
                W += -eta*grad_W

        train_loss = ComputeCost(train_X, train_Y, W, _lambda, b, loss=loss)
        train_acc = ComputeAccuracy(train_X, train_y, W, b, loss)

        val_loss = ComputeCost(val_X, val_Y, W, _lambda, b, loss=loss)
        val_acc = ComputeAccuracy(val_X, val_y, W, b, loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # print(f"Epoch {epoch+1}: loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")

        ## bonus 2:
        if early_stopping:
            if(val_loss-history['val_loss'][epoch-1] > loss_threshold):
                print("val loss increased a lot")
                print(f"It was {history['val_loss'][epoch-1]}, now it is {val_loss}")
                break

        # if bonus4:
        if weight_decay:
            eta *= 0.9

    return W, b, history


def plot_and_save_learning_curve(history, name):

    train_acc = history['train_acc']
    val_acc = history['val_acc']

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig.suptitle('Plots for these paramerers')

    ax1.plot(train_acc, label='Train')
    ax1.plot(val_acc, label='Validation')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(train_loss, label='Train')
    ax2.plot(val_loss, label='Validation')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # plt.show()
    plt.savefig(name)


if __name__ == "__main__":
    # all training data and small val data
    bonus1 = True

    # longer training (more epochs) and early stopping
    bonus2 = True

    # weight decay
    bonus4 = True

    # 'svm' loss or 'cross_entropy'
    bonus_svm = True

    losses = ('cross_entropy', 'svm')
    if bonus_svm:
        loss = losses[1]
    else:
        loss = losses[0]

    if bonus1:
        train_X1, train_Y1, train_y1 = _LoadBatch('cifar-10-batches-py/data_batch_1')
        train_X2, train_Y2, train_y2 = _LoadBatch('cifar-10-batches-py/data_batch_2')
        train_X3, train_Y3, train_y3 = _LoadBatch('cifar-10-batches-py/data_batch_3')
        train_X4, train_Y4, train_y4 = _LoadBatch('cifar-10-batches-py/data_batch_4')
        train_X5, train_Y5, train_y5 = _LoadBatch('cifar-10-batches-py/data_batch_5')

        train_X = np.hstack((train_X1, train_X2, train_X3, train_X4, train_X5))
        train_Y = np.hstack((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5))
        train_y = np.array(train_y1+train_y2+train_y3+train_y4+train_y5)

        indexes = np.random.choice(train_X.shape[1], 1000, replace=False)

        val_X = train_X[:, indexes]
        val_Y = train_Y[:, indexes]
        val_y = train_y[indexes]

        train_X = np.delete(train_X, indexes, 1)
        train_Y = np.delete(train_Y, indexes, 1)
        train_y = np.delete(train_y, indexes)

        test_X, test_Y, test_y = _LoadBatch('cifar-10-batches-py/test_batch')

    else:
        train_X, train_Y, train_y = _LoadBatch('cifar-10-batches-py/data_batch_1')
        val_X, val_Y, val_y = _LoadBatch('cifar-10-batches-py/data_batch_2')
        test_X, test_Y, test_y = _LoadBatch('cifar-10-batches-py/test_batch')

    # train_X = normalize(train_X)
    # val_X = normalize(val_X)
    # test_X = normalize(test_X)

    train_X,val_X,test_X = normalize(train_X, val_X, test_X)

    testing = False
    if testing:
        smaller_set = True
        if smaller_set:
            print("Testing small set without regularization:", end='\n\n')
            X = train_X[0:20, [0]]
            Y = train_Y[:, [0]]
            y = train_y[0]
            W = W[:, :20]
            b = b
            _lambda = 0.0
        else:
            print("Testing bigger set with regularization:", end='\n\n')
            X = train_X[:, :5]
            Y = train_Y[:, :5]
            y = train_y[:5]
            W = W
            b = b
            _lambda = 0.01
        P = EvaluateClassifier(X, W, b)
        testing_gradients(X, Y, y, P, W, b, _lambda)

    # Define the network parameters
    parameters = [
        {
            '_lambda': 0,
            'GDparams': {
                'n_epochs': 40,
                'n_batch': 100,
                'eta': 0.1
            }
        },
        {
            '_lambda': 0,
            'GDparams': {
                'n_epochs': 40,
                'n_batch': 100,
                'eta': 0.001
            }
        },
        {
            '_lambda': 0.1,
            'GDparams': {
                'n_epochs': 40,
                'n_batch': 100,
                'eta': 0.001
            }
        },
        {
            '_lambda': 1.0,
            'GDparams': {
                'n_epochs': 40,
                'n_batch': 100,
                'eta': 0.001
            }
        }
    ]

    if bonus2:
        for param in parameters:
            param['GDparams'].update({'n_epochs': 100})

    for i,param in enumerate(parameters):
        W, b = initialize_weights()
        print("Starting with", str(param))
        _lambda = param['_lambda']
        GDparams = param['GDparams']

        if bonus2 and bonus4: Wstar, bstar, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, early_stopping=True, weight_decay=True, loss=loss)
        elif bonus2: Wstar, bstar, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, early_stopping=True, loss=loss)
        elif bonus4: Wstar, bstar, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, weight_decay=True, loss=loss)
        else: Wstar, bstar, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, loss=loss)
        
        print("The test accuracy is",ComputeAccuracy(test_X, test_y, Wstar))

        # name_learning = f"epochs={GDparams['n_epochs']},batch={GDparams['n_batch']},lr={GDparams['eta']},lambda={_lambda}_learning.png"
        # name_weight = f"epochs={GDparams['n_epochs']},batch={GDparams['n_batch']},lr={GDparams['eta']},lambda={_lambda}_weight.png"
        
        name_learning = f"case_{str(i)}"
        name_weight = f"case_w_{str(i)}"

        plot_and_save_learning_curve(history, name_learning)
        montage(Wstar, name_weight)
