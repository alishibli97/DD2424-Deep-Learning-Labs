import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
from numba import jit

"""
functions
"""

num_labels = 10

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Datasets/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def _LoadBatch(file):
    data = LoadBatch(file)
    X = data[b'data'].T/255
    y = data[b'labels']

    # one hot encoding
    Y = np.zeros(shape=(num_labels, len(y)))
    for i, label in enumerate(y):
        Y[label, i] = 1

    return X, Y, y

def normalize(train_X, val_X, test_X):
    std = train_X.std(axis=1).reshape(train_X.shape[0], 1)
    mean = train_X.mean(axis=1).reshape(train_X.shape[0], 1)
    
    train_X = (train_X-mean)/std
    val_X = (val_X-mean)/std
    test_X = (test_X-mean)/std

    return train_X, val_X, test_X

def initialize_weights(input_dimension, hidden_dimension, output_dimension):
    np.random.seed(0)
    W1 = np.random.normal(size=(hidden_dimension, input_dimension), loc=0, scale=1/np.sqrt(input_dimension))
    W2 = np.random.normal(size=(output_dimension, hidden_dimension), loc=0, scale=1/np.sqrt(hidden_dimension))
    b1 = np.zeros(shape=(hidden_dimension,1))
    b2 = np.zeros(shape=(output_dimension,1))
    return W1, b1, W2, b2

def relu(S):
    H = S
    H[H<0] = 0
    return H

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def EvaluateClassifier(X, W1, b1, W2, b2):
    S1 = np.dot(W1,X)+b1
    H = relu(S1)
    S = np.dot(W2,H)+b2
    P = softmax(S)
    return P, H

def ComputeCost(X, Y, W1, b1, W2, b2, _lambda):
    # Compute the predictions
    P, H = EvaluateClassifier(X, W1, b1, W2, b2)
    
    # Compute the loss function term
    loss_cross = sum(-np.log((Y*P).sum(axis=0)))
    
    # Compute the regularization term
    loss_regularization = _lambda*((W1**2).sum()+(W2**2).sum())
    
    # Sum the total cost
    J = loss_cross/X.shape[1]+loss_regularization
    return J

def ComputeAccuracy(X, y, W1, b1, W2, b2):
    # Compute the predictions
    P, H = EvaluateClassifier(X, W1, b1, W2, b2)
    
    # Compute the accuracy
    acc = np.mean(y==np.argmax(P, 0))
    return acc

def ComputeGradients(X, Y, P, H, W1, W2, _lambda):
    n = X.shape[1]
    C = Y.shape[0]
    M = H.shape[0]
    G = -(Y-P)
    grad_W2 = (1/n)*np.dot(G,H.T)+2*_lambda*W2
    grad_b2 = (1/n)*np.dot(G,np.ones(shape=(n,1))).reshape(C, 1)
    G = np.dot(W2.T,G)
    G = G*(H>0)
    grad_W1 = (1/n)*np.dot(G,X.T)+2*_lambda*W1
    grad_b1 = (1/n)*(np.dot(G,np.ones(shape=(n,1)))).reshape(M, 1)
    return grad_W2, grad_b2, grad_W1, grad_b1

def ComputeGradsNum(X, Y, W1, b1, W2, b2, _lambda, h=1e-5):
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b2 = np.zeros(shape=b2.shape)
    grad_W1 = np.zeros(shape=W1.shape)
    grad_b1 = np.zeros(shape=b1.shape)   
    c = ComputeCost(X, Y, W1, b1, W2, b2, _lambda)
    
    for i in range(b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i,0] = b1_try[i,0]+h
        c2 = ComputeCost(X, Y, W1, b1_try, W2, b2, _lambda)
        grad_b1[i,0] = (c2-c)/h
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i,j] = W1_try[i,j]+h
            c2 = ComputeCost(X, Y, W1_try, b1, W2, b2, _lambda)
            grad_W1[i,j] = (c2-c)/h
    
    for i in range(b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i,0] = b2_try[i,0]+h
        c2 = ComputeCost(X, Y, W1, b1, W2, b2_try, _lambda)
        grad_b2[i,0] = (c2-c)/h
    
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i,j] = W2_try[i,j]+h
            c2 = ComputeCost(X, Y, W1, b1, W2_try, b2, _lambda)
            grad_W2[i,j] = (c2-c)/h
    
    return grad_W2, grad_b2, grad_W1, grad_b1

def ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, _lambda, h=1e-5):
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b2 = np.zeros(shape=b2.shape)
    grad_W1 = np.zeros(shape=W1.shape)
    grad_b1 = np.zeros(shape=b1.shape)

    for i in range(b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i] -= h
        c1 = ComputeCost(X, Y, W1, b1_try, W2, b2, _lambda)
        
        b1_try = b1.copy()
        b1_try[i] += h
        c2 = ComputeCost(X, Y, W1, b1_try, W2, b2, _lambda)
        
        grad_b1[i] = (c2-c1) / (2*h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i,j] += h
            c1 = ComputeCost(X, Y, W1_try, b1, W2, b2, _lambda)
            
            W1_try = W1.copy()
            W1_try[i,j] -= h
            c2 = ComputeCost(X, Y, W1_try, b1, W2, b2, _lambda)
            
            grad_W1[i, j] = (c2-c1) / (2*h)

    for i in range(b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i] -= h
        c1 = ComputeCost(X, Y, W1, b1, W2, b2_try, _lambda)
        
        b2_try = b2.copy()
        b2_try[i] += h
        c2 = ComputeCost(X, Y, W1, b1, W2, b2_try, _lambda)
        
        grad_b2[i] = (c2-c1) / (2*h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i,j] += h
            c1 = ComputeCost(X, Y, W1, b1, W2_try, b2, _lambda)
            
            W2_try = W2.copy()
            W2_try[i,j] -= h
            c2 = ComputeCost(X, Y, W1, b1, W2_try, b2, _lambda)
            
            grad_W2[i, j] = (c2-c1) / (2*h)
    
    return grad_W2, grad_b2, grad_W1, grad_b1

def testing_gradients(X, Y, y, P, H, W1, b1, W2, b2, _lambda, h=1e-5):

    grad_w2_analytical, grad_b2_analytical, grad_w1_analytical, grad_b1_analytical = ComputeGradients(X, Y, P, H, W1, W2, _lambda)
    grad_w2_num, grad_b2_num, grad_w1_num, grad_b1_num = ComputeGradsNum(X, Y, W1, b1, W2, b2, _lambda, h=1e-5)
    grad_w2_num_slow, grad_b2_num_slow, grad_w1_num_slow, grad_b1_num_slow = ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, _lambda, h=1e-5)

    epsilon = 1e-6

    diff_w1_computed_num = np.abs(grad_w1_num-grad_w1_analytical)
    diff_b1_computed_num = np.abs(grad_b1_num-grad_b1_analytical)
    diff_w1_computed_num_slow = np.abs(grad_w1_num_slow-grad_w1_analytical)
    diff_b1_computed_num_slow = np.abs(grad_b1_num_slow-grad_b1_analytical)

    diff_w2_computed_num = np.abs(grad_w2_num-grad_w2_analytical)
    diff_b2_computed_num = np.abs(grad_b2_num-grad_b2_analytical)
    diff_w2_computed_num_slow = np.abs(grad_w2_num_slow-grad_w2_analytical)
    diff_b2_computed_num_slow = np.abs(grad_b2_num_slow-grad_b2_analytical)

    error_w1_num = np.sum(diff_w1_computed_num > 1e-6)
    error_w1_num_acc = np.mean(diff_w1_computed_num > 1e-6)
    error_b1_num = np.sum(diff_b1_computed_num > 1e-6)
    error_b1_num_acc = np.mean(diff_b1_computed_num > 1e-6)
    error_w1_num_slow = np.sum(diff_w1_computed_num_slow > 1e-6)
    error_w1_num_slow_acc = np.mean(diff_w1_computed_num_slow > 1e-6)
    error_b1_num_slow = np.sum(diff_b1_computed_num_slow > 1e-6)
    error_b1_num_slow_acc = np.mean(diff_b1_computed_num_slow > 1e-6)

    error_w2_num = np.sum(diff_w2_computed_num > 1e-6)
    error_w2_num_acc = np.mean(diff_w2_computed_num > 1e-6)
    error_b2_num = np.sum(diff_b2_computed_num > 1e-6)
    error_b2_num_acc = np.mean(diff_b2_computed_num > 1e-6)
    error_w2_num_slow = np.sum(diff_w2_computed_num_slow > 1e-6)
    error_w2_num_slow_acc = np.mean(diff_w2_computed_num_slow > 1e-6)
    error_b2_num_slow = np.sum(diff_b2_computed_num_slow > 1e-6)
    error_b2_num_slow_acc = np.mean(diff_b2_computed_num_slow > 1e-6)

    error_rel_1 = diff_w1_computed_num / np.maximum(epsilon, np.abs(grad_w1_num)+np.abs(grad_w1_analytical))
    error_rel_2 = diff_b1_computed_num / np.maximum(epsilon, np.abs(grad_b1_num)+np.abs(grad_b1_analytical))
    error_rel_3 = diff_w1_computed_num_slow / np.maximum(epsilon, np.abs(grad_w1_num_slow)+np.abs(grad_w1_analytical))
    error_rel_4 = diff_b1_computed_num_slow / np.maximum(epsilon, np.abs(grad_b1_num_slow)+np.abs(grad_b1_analytical))

    error_rel_5 = diff_w2_computed_num / np.maximum(epsilon, np.abs(grad_w2_num)+np.abs(grad_w2_analytical))
    error_rel_6 = diff_b2_computed_num / np.maximum(epsilon, np.abs(grad_b2_num)+np.abs(grad_b2_analytical))
    error_rel_7 = diff_w2_computed_num_slow / np.maximum(epsilon, np.abs(grad_w2_num_slow)+np.abs(grad_w2_analytical))
    error_rel_8 = diff_b2_computed_num_slow / np.maximum(epsilon, np.abs(grad_b2_num_slow)+np.abs(grad_b2_analytical))

    error_relative_w1_num = np.sum(error_rel_1 > epsilon)
    error_relative_w1_num_acc = np.mean(error_rel_1 > epsilon)
    error_relative_b1_num = np.sum(error_rel_2 > epsilon)
    error_relative_b1_num_acc = np.mean(error_rel_2 > epsilon)
    error_relative_w1_num_slow = np.sum(error_rel_3 > epsilon)
    error_relative_w1_num_slow_acc = np.mean(error_rel_3 > epsilon)
    error_relative_b1_num_slow = np.sum(error_rel_4 > epsilon)
    error_relative_b1_num_slow_acc = np.mean(error_rel_4 > epsilon)

    error_relative_w2_num = np.sum(error_rel_5 > epsilon)
    error_relative_w2_num_acc = np.mean(error_rel_5 > epsilon)
    error_relative_b2_num = np.sum(error_rel_6 > epsilon)
    error_relative_b2_num_acc = np.mean(error_rel_6 > epsilon)
    error_relative_w2_num_slow = np.sum(error_rel_7 > epsilon)
    error_relative_w2_num_slow_acc = np.mean(error_rel_7 > epsilon)
    error_relative_b2_num_slow = np.sum(error_rel_8 > epsilon)
    error_relative_b2_num_slow_acc = np.mean(error_rel_8 > epsilon)

    print("Using error as number of points where difference is less than epsilon = 1e-6", end='\n\n')

    print("Between ComputeGradients and ComputeGradNum:")
    print(f"There are {error_w1_num} errors for weights 1 (error rate = {error_w1_num_acc*100}%)")
    print(f"There are {error_b1_num} errors for biases 1 (error rate = {error_b1_num_acc*100}%)")
    print(f"There are {error_w2_num} errors for weights 2 (error rate = {error_w2_num_acc*100}%)")
    print(f"There are {error_b2_num} errors for biases 2 (error rate = {error_b2_num_acc*100}%)")

    print()

    print("Between ComputeGradients and ComputeGradNumSlow:")
    print(f"There are {error_w1_num_slow} errors for weights 1 (error rate = {error_w1_num_slow_acc*100}%)")
    print(f"There are {error_b1_num_slow} errors for biases 1 (error rate = {error_b1_num_slow_acc*100}%)")
    print(f"There are {error_w2_num_slow} errors for weights 2 (error rate = {error_w2_num_slow_acc*100}%)")
    print(f"There are {error_b2_num_slow} errors for biases 2 (error rate = {error_b2_num_slow_acc*100}%)")

    print()

    print("Using relative error as number of points where difference is less than epsilon = 1e-6", end='\n\n')

    print("Between ComputeGradients and ComputeGradNum:")
    print(f"There are {error_relative_w1_num} errors for weights 1 (error rate = {error_relative_w1_num_acc*100}%)")
    print(f"There are {error_relative_b1_num} errors for biases 1 (error rate = {error_relative_b1_num_acc*100}%)")
    print(f"There are {error_relative_w2_num} errors for weights 2 (error rate = {error_relative_w2_num_acc*100}%)")
    print(f"There are {error_relative_b2_num} errors for biases 2 (error rate = {error_relative_b2_num_acc*100}%)")

    print()

    print("Between ComputeGradients and ComputeGradNumSlow:")
    print(f"There are {error_relative_w1_num_slow} errors for weights (error rate = {error_relative_w1_num_slow_acc}%)")
    print(f"There are {error_relative_b1_num_slow} errors for biases (error rate = {error_relative_b1_num_slow_acc}%)")
    print(f"There are {error_relative_w2_num_slow} errors for weights (error rate = {error_relative_w2_num_slow_acc}%)")
    print(f"There are {error_relative_b2_num_slow} errors for biases (error rate = {error_relative_b2_num_slow_acc}%)")

def MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, _lambda,dropout_rate=0.0,add_noise=False, smith=False):
    n = train_X.shape[1]
    # eta = GDparams['eta']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']
    n_batch = GDparams['n_batch']
    n_cycles = GDparams['n_cycles']

    # cycle_length = 2*n_s

    loss_threshold = 0.4

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'etas': [],
    }

    # dict for nodes to drop per batch
    nodes_dropout = {}

    batches = []
    max_index = n//n_batch
    for j in range(max_index): # n_s
        j_start = j*n_batch
        j_end = (j+1)*n_batch
        inds = range(j_start, j_end)

        X_batch = train_X[:, inds]
        Y_batch = train_Y[:, inds]
        y_batch = [train_y[index] for index in inds]

        batches.append([X_batch,Y_batch,y_batch])

        nodes_dropout[j] = np.random.rand(W1.shape[0], n_batch) < dropout_rate

    index = 0
    exit=False
    for cycle in range(n_cycles):
        for t in range(2*cycle*n_s, 2*(cycle+1)*n_s):
            if smith and t==n_s:
                exit=True
                break
            if (2*cycle*n_s <= t < (2*cycle+1)*n_s):
                eta = eta_min+(t-2*cycle*n_s)/n_s*(eta_max-eta_min)
            elif ((2*cycle+1)*n_s <= t < 2*(cycle+1)*n_s):
                eta = eta_max-(t-(2*cycle+1)*n_s)/n_s*(eta_max-eta_min)

            X_batch, Y_batch, y_batch = batches[index]
            if add_noise: X_batch += np.random.normal(size=X_batch.shape, loc=0, scale=0.1)

            P_batch, H_batch = EvaluateClassifier(X_batch, W1, b1, W2, b2)

            # add condition is we have dropout or not before multiplying
            H_batch = H_batch*nodes_dropout[index]

            grad_W2, grad_b2, grad_W1, grad_b1 = ComputeGradients(X_batch, Y_batch, P_batch, H_batch, W1, W2, _lambda)

            W1 += -eta*grad_W1
            b1 += -eta*grad_b1
            W2 += -eta*grad_W2
            b2 += -eta*grad_b2

            index+=1
            if index>=max_index:
                index = 0

            train_loss = ComputeCost(train_X, train_Y, W1, b1, W2, b2, _lambda) # loss=loss
            train_acc = ComputeAccuracy(train_X, train_y, W1, b1, W2, b2) # loss

            val_loss = ComputeCost(val_X, val_Y, W1, b1, W2, b2, _lambda) # loss=loss
            val_acc = ComputeAccuracy(val_X, val_y, W1, b1, W2, b2) # loss

            # if t%100==0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['etas'].append(eta)

            sys.stdout.write(f"\rIter {t+1}/{2*n_s*n_cycles}: loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")
            sys.stdout.flush()
        if exit: break
        #print(f"Cycle {cycle+1}: loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")
    print()

    return W1, b1, W2, b2, history

def exhaustivesearch():
    lambda_best = 0.004

    # first optimizing to get best number of cycles

    GDparams = {
        'eta_min':1e-5,
        'eta_max':1e-1,
        'n_batch':100,
        'n_cycles':10
    }
    GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
    print(f"The cycle width being used is {GDparams['n_s']} steps/cycle") # 
    W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)    
    W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best)
    print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
    name = "Learning_curve_for_10_cycles.png"
    plot_and_save_learning_curve(history, name=name)

# @jit
def check_hidden_dimensions(hidden_dimensions):
    test_accuracies = []
    histories = []
    for i,hidden_dimension in enumerate(hidden_dimensions):
        # print(f"Checking for {hidden_dimension} hidden units")
        lambda_best = 0.004
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':500,
            'n_cycles':2
            }
        GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
        W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
        W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best)
        
        test_acc = ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check)
        
        test_accuracies.append(test_acc)
        histories.append(history)
        
        # print("The test accuracy is",test_acc)
        # name = "Learning_curves_for_varying_hidden_dimension.png"
        # plot_and_save_learning_curve(history, name=name,save_fig= i==len(hidden_dimensions)-1)
    return test_accuracies,histories

def plot_and_save_learning_curve(history, name, xlabel='update step', varying_lambda=False, smith=False):

    train_acc = history['train_acc']
    val_acc = history['val_acc']

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    if not smith:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, (ax1) = plt.subplots(1, 1)

    # fig.suptitle('Plots for these paramerers')

    if not varying_lambda:
        if smith:
            ax1.plot(history['etas'], train_acc, label='Train')
            ax1.plot(history['etas'], val_acc, label='Validation')

        else:
            ax1.plot(train_acc, label='Train')
            ax1.plot(val_acc, label='Validation')
            
            ax2.plot(train_loss, label='Train')
            ax2.plot(val_loss, label='Validation')
        
    else:
        ax1.plot(history['log lambda'],train_acc, label='Train')
        ax1.plot(history['log lambda'],val_acc, label='Validation')

        ax2.plot(history['log lambda'],train_loss, label='Train')
        ax2.plot(history['log lambda'],val_loss, label='Validation')

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.legend()

    if not smith:
        ax2.set_ylabel('Loss')
        ax2.set_xlabel(xlabel)
        ax2.legend()

    # plt.show()
    fig.tight_layout(pad=3.0)
    plt.savefig(name)


def plot(histories,hidden_dimensions):
    name = "Learning_curves_for_varying_hidden_dimension.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    for  i,history in enumerate(histories):
        train_acc = history['train_acc']
        val_acc = history['val_acc']

        train_loss = history['train_loss']
        val_loss = history['val_loss']

        x_range = np.array(range(len(train_acc)))*100

        ax1.plot(x_range,train_acc, label=f'{hidden_dimensions[i]} hidden dimensions - train')
        ax1.plot(x_range,val_acc, label=f'{hidden_dimensions[i]} hidden dimensions - validation', linestyle="--")
        
        ax2.plot(x_range, train_loss, label=f'{hidden_dimensions[i]} hidden dimensions - train')
        ax2.plot(x_range, val_loss, label=f'{hidden_dimensions[i]} hidden dimensions - validation', linestyle="--")

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel("update step")
    ax1.legend()

    ax2.set_ylabel('Loss')
    ax2.set_xlabel("update step")
    ax2.legend()

    # fig.tight_layout(pad=3.0)
    plt.savefig(name)

if __name__ == "__main__":
    train_X1, train_Y1, train_y1 = _LoadBatch('cifar-10-batches-py/data_batch_1')
    train_X2, train_Y2, train_y2 = _LoadBatch('cifar-10-batches-py/data_batch_2')
    train_X3, train_Y3, train_y3 = _LoadBatch('cifar-10-batches-py/data_batch_3')
    train_X4, train_Y4, train_y4 = _LoadBatch('cifar-10-batches-py/data_batch_4')
    train_X5, train_Y5, train_y5 = _LoadBatch('cifar-10-batches-py/data_batch_5')

    train_X = np.hstack((train_X1, train_X2, train_X3, train_X4, train_X5))
    train_Y = np.hstack((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5))
    train_y = np.array(train_y1+train_y2+train_y3+train_y4+train_y5)

    best_result = True
    val_size = 1000 if best_result else 5000

    indexes = np.random.choice(train_X.shape[1], val_size, replace=False)

    val_X = train_X[:, indexes]
    val_Y = train_Y[:, indexes]
    val_y = train_y[indexes]

    # val_X, val_Y, val_y = _LoadBatch('cifar-10-batches-py/data_batch_2')
    # test_X, test_Y, test_y = _LoadBatch('cifar-10-batches-py/test_batch')

    test_X, test_Y, test_y = _LoadBatch('cifar-10-batches-py/test_batch')

    train_X,val_X,test_X = normalize(train_X, val_X, test_X)

    input_dimension = train_X.shape[0]
    output_dimension = train_Y.shape[0]

    bonus_exhaustive_search = False
    bonus_hiddendimensions = False # done
    bonus_dropout = False # done
    bonus_noise = True # done
    bonus_smith = False # done
    
    if bonus_exhaustive_search:
        print("Exhaustive search")
        hidden_dimension = 50
        exhaustivesearch()


    elif bonus_hiddendimensions:
        print("More hidden nodes")

        hidden_dimensions = [50,100,150,200]
        test_accuracies,histories = check_hidden_dimensions(hidden_dimensions)
        print(test_accuracies) # [0.4911, 0.5079, 0.5156, 0.5204]
        plot(histories,hidden_dimensions)

    elif bonus_dropout:
        print("Dropout")

        lambda_best = 0.004
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':200,
            'n_cycles':3
            }
        GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
        hidden_dimension = 50
        W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
        W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best,dropout_rate=0.7)
        print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check)) # 0.3:0.4174, 0.5:0.477, 0.7:0.4936
        name = "Default parameters with dropout.png"
        plot_and_save_learning_curve(history, name=name)

    elif bonus_noise:
        print("Add noise")
        
        lambda_best = 0.004
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':200,
            'n_cycles':3
            }
        GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
        hidden_dimension = 50
        W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
        W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best,add_noise=True,dropout_rate=0.7)
        print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check)) # The test accuracy is 0.4934
        name = "Added noise.png"
        plot_and_save_learning_curve(history, name=name)

    elif bonus_smith:
        print("Smith")

        finding_best_vals = False
        if finding_best_vals:

            lambda_best = 0.004
            GDparams = {
                'eta_min':1e-5,
                'eta_max':1e-1,
                'n_batch':200,
                'n_cycles':3
                }
            GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
            hidden_dimension = 50
            W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
            W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best, dropout_rate=0.7, smith=True)
            print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check)) # 0.3:0.4174, 0.5:0.477, 0.7:0.4936
            name = "smith_finding_etas.png"
            plot_and_save_learning_curve(history, name=name, xlabel="Learning rate", smith=True)
        else:
            lambda_best = 0.004
            GDparams = {
                'eta_min':0.04,
                'eta_max':0.1,
                'n_batch':200,
                'n_cycles':3
                }
            GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
            hidden_dimension = 50
            W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
            W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best, dropout_rate=0.7)
            print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
            name = "results_smith.png"
            plot_and_save_learning_curve(history, name=name)

    else:
        hidden_dimension = 50
        testing_gradient = False
        if testing_gradient:
            X = train_X1[:20, [0]]
            Y = train_Y1[:, [0]]
            y = train_y[0]

            W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)

            W1_test = W1[:,:20]
            b1_test = b1
            W2_test = W2
            b2_test = b2

            P,H = EvaluateClassifier(X, W1_test, b1_test, W2_test, b2_test)

            testing_gradients(X, Y, y, P, H, W1_test, b1_test, W2_test, b2_test, _lambda=0, h=1e-5)
        else:

            smaller_params = False

            if smaller_params:
                W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
                
                # X, Y, y = train_X[:,0:100], train_Y[:,0:100], train_y[0:100]

                # define network parameters
                _lambda = 0.01
                GDparams = {
                    'eta_min':1e-5,
                    'eta_max':1e-1,
                    'n_s':500,
                    'n_batch':100, 
                    'n_cycles':1
                    }

                # Train the network
                W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, _lambda)

                print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
                name_learning = "figure 3"
                plot_and_save_learning_curve(history, name_learning)

            else:

                lambda_optimizing = True

                if not lambda_optimizing:

                    W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
                    
                    # define network parameters
                    _lambda = 0.01
                    GDparams = {
                        'eta_min':1e-5,
                        'eta_max':1e-1,
                        'n_s':800,
                        'n_batch':100, 
                        'n_cycles':3
                        }

                    # Train the network
                    W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, _lambda)

                    print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
                    name_learning = "figure 4"
                    plot_and_save_learning_curve(history, name_learning)
                
                else:

                    if not best_result:
                        coarse = False

                        if coarse:
                            l_min, l_max = -5, -1
                            l = l_min+(l_max-l_min)*np.random.rand(8)
                            lambdas_coarse = list(10**l)
                            lambdas_coarse.sort()
                            lambdas_list = lambdas_coarse.copy()

                            # best coarse value
                            lambda_best = lambdas_list[1]
                        
                        else:
                            lambdas_fine = np.arange(0, 0.008, 0.001)
                            lambdas_list = lambdas_fine.copy()

                        GDparams = {
                            'eta_min':1e-5,
                            'eta_max':1e-1,
                            'n_batch':500, # 100
                            'n_cycles':2
                            }
                        GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])

                        metrics_lambdas = {
                            'lambda':[],
                            'train_loss':[],
                            'train_acc':[],
                            'val_loss':[],
                            'val_acc':[]
                        }

                        for _lambda in lambdas_list:
                            print(f"Checking lambda={_lambda}")

                            W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
                            W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, _lambda)
                            
                            metrics_lambdas['lambda'].append(np.log(_lambda))
                            
                            metrics_lambdas['train_acc'].append(history['train_acc'][-1])
                            metrics_lambdas['val_acc'].append(history['val_acc'][-1])
                            
                            metrics_lambdas['train_loss'].append(history['train_loss'][-1])
                            metrics_lambdas['val_loss'].append(history['val_loss'][-1])

                            print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
                        
                        if coarse: name = 'Learning curve - coarse search'
                        else: name = 'Learning curve - fine search'
                        plot_and_save_learning_curve(metrics_lambdas, name=name, xlabel='lambda', varying_lambda=True)
                    
                    else: # best result
                        lambda_best = 0.004
                        GDparams = {
                            'eta_min':1e-5,
                            'eta_max':1e-1,
                            'n_batch':100,
                            'n_cycles':3
                        }
                        GDparams['n_s'] = int(2*train_X.shape[1]/GDparams['n_batch'])
                        W1, b1, W2, b2 = initialize_weights(input_dimension, hidden_dimension, output_dimension)
                        W1_check, b1_check, W2_check, b2_check, history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W1, b1, W2, b2, lambda_best)
                        print("The test accuracy is",ComputeAccuracy(test_X, test_y, W1_check, b1_check, W2_check, b2_check))
                        name = "Learning_curve_for_best_lambda=0.004.png"
                        plot_and_save_learning_curve(history, name=name)