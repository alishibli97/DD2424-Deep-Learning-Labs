import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


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

def initialize_weights(input_dimension, hidden_dimensions, output_dimension,He=True,sig=None,batch_norm=False):
    np.random.seed(0)
    k = len(hidden_dimensions)+1
    W = []
    b = []
    hidden_dimensions = [input_dimension]+hidden_dimensions+[output_dimension]
    gamma = []
    beta = []
    for i in range(k):
        if He:
            W.append(np.random.normal(size=(hidden_dimensions[i+1],hidden_dimensions[i]),loc=0,scale=1/np.sqrt(hidden_dimensions[i])))
            b.append(np.zeros(shape=(hidden_dimensions[i+1],1)))
        else: # normal distribution with given sig
            W.append(np.random.normal(size=(hidden_dimensions[i+1],hidden_dimensions[i]),loc=0,scale=sig))
            b.append(np.zeros(shape=(hidden_dimensions[i+1],1)))
        if i<(k-1):
            gamma.append(np.ones(shape=(hidden_dimensions[i+1],1)))
            beta.append(np.zeros(shape=(hidden_dimensions[i+1],1)))
    if batch_norm: return W,b,gamma,beta
    else: return W, b

def relu(S):
    H = S
    H[H<0] = 0
    return H

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def BatchNormalize(s,mean,var):
    return (s-mean)/(np.sqrt(var+1e-15))

def EvaluateClassifier(X, W, b, gamma=None, beta=None, batch_norm=False, means=None, vars=None, norm_before=True):
    k = len(W)
    S = []
    S_normalized = []
    if not means and not vars:
        means = []
        vars = []
        add_means_vars = True
    X_batch = [X.copy()]
    for i in range(k-1):
        s = np.dot(W[i],X_batch[i])+b[i]; S.append(s)
        if batch_norm:
            if add_means_vars:
                mean = s.mean(axis=1).reshape(-1,1); means.append(mean)
                var = s.var(axis=1).reshape(-1,1); vars.append(var)
            else:
                mean = means[i]
                var = vars[i]
            if norm_before: # apply batch norm before nonlinear activation function
                s_normalized = BatchNormalize(s, mean, var); S_normalized.append(s_normalized)
                S_scaled = gamma[i]*s_normalized + beta[i]
                X_batch.append(relu(S_scaled))
            else: # apply batch norm after nonlinear activation function
                s_normalized = BatchNormalize(relu(s), mean, var); S_normalized.append(s_normalized)
                S_scaled = gamma[i]*s_normalized + beta[i]
                X_batch.append(S_scaled)
        else:
            X_batch.append(relu(s))
    P_batch = softmax(np.dot(W[k-1],X_batch[k-1])+b[k-1])
    if batch_norm: return P_batch, X_batch[1:], S,S_normalized, means, vars
    else: return P_batch, X_batch[1:]

def ComputeCost(X, Y, _lambda, W, b, gamma=None, beta=None, batch_norm=False, means=None, vars=None):
    if batch_norm: 
        if not means and not vars: P, S_BN, S, X_layers, mean, var = EvaluateClassifier(X, W, b, gamma=gamma, beta=beta, batch_norm=batch_norm, means=means, vars=vars)
        else: P, S_BN, S, X_layers, mean, var = EvaluateClassifier(X, W, b, gamma=gamma, beta=beta, batch_norm=batch_norm)
    else: P, X_layers = EvaluateClassifier(X, W, b)
    loss_cross = sum(-np.log((Y*P).sum(axis=0)))
    loss_regularization = 0
    for W_l in W:
        loss_regularization += _lambda*((W_l**2).sum())
    J = loss_cross/X.shape[1]+loss_regularization    
    return J

def ComputeAccuracy(X, y, W, b, gamma=None, beta=None, batch_norm=False, means=None, vars=None):
    if batch_norm: 
        if not means and not vars: P_batch, X_batch, S,S_normalized, means, vars = EvaluateClassifier(X, W, b,gamma,beta,batch_norm,means,vars)
        else: P_batch, X_batch, S,S_normalized, means, vars = EvaluateClassifier(X, W, b,gamma,beta,batch_norm)
    else: P_batch, X_batch = EvaluateClassifier(X, W, b)
    return np.mean(y==np.argmax(P_batch, 0))

def BatchNormBackPass(G, S, mean, var):
    n = S.shape[1]
    epsilon = 1e-10
    sigma_1 = (var+epsilon)**(-0.5)
    sigma_2 = (var+epsilon)**(-1.5)
    G1 = G*np.dot(sigma_1,np.ones((1,n)))
    G2 = G*np.dot(sigma_2,np.ones((1,n)))
    D = S-np.dot(mean,np.ones((1,n)))
    c = np.dot((G2*D),np.ones((n,1)))
    G_normalized = G1-(1/n)*(np.dot(G1,np.ones((n,1))))-(1/n)*D*(np.dot(c,np.ones((1,n))))
    return G_normalized

def ComputeGradients(X, Y, P, X_batch, W, _lambda, means=None, vars=None,S=None,S_BN=None,gamma=None,beta=None,batch_norm=False):
    k = len(W)
    n_b = X.shape[1]

    grad_W = [None]*k
    grad_b = [None]*k
    if batch_norm:
        grad_gamma = [None]*(k-1)
        grad_beta = [None]*(k-1)

    # Propagate the gradient through the loss and softmax operations
    G = -(Y-P)
    X_batch = [X.copy()]+X_batch

    # The gradients of J w.r.t. bias vector b_k and W_k
    grad_W[k-1] = (1/n_b)*np.dot(G,X_batch[k-1].T) + 2*_lambda*W[k-1]
    grad_b[k-1] = (1/n_b)*np.dot(G,np.ones(shape=(n_b,1)))

    # Propagate the gradient through the loss and softmax operations
    G = np.dot(W[k-1].T,G)
    G = G*(X_batch[k-1]>0)

    for l in range(k-2,-1,-1):
        if batch_norm:
            # Compute gradient for the scale and offset parameters for layer l:
            grad_gamma[l] = (1/n_b)*np.dot((G*S_BN[l]),np.ones((n_b,1)))
            grad_beta[l] = (1/n_b)*np.dot(G,np.ones((n_b,1)))

            # Propagate the gradients through the scale and shift
            G = G*(np.dot(gamma[l],np.ones((1,n_b))))

            # Propagate G batch through the batch normalization
            G = BatchNormBackPass(G, S[l], means[l], vars[l])

        # The gradients of J w.r.t. bias vector b_l and W_l
        grad_W[l] = (1/n_b)*np.dot(G,X_batch[l].T) + 2*_lambda*W[l]
        grad_b[l] = (1/n_b)*np.dot(G,np.ones(shape=(n_b,1)))
        
        # If l > 1 propagate G batch to the previous layer
        if l>0:
            G = np.dot(W[l].T,G)
            G = G*(X_batch[l]>0)

    if batch_norm: return grad_W, grad_b, grad_gamma, grad_beta
    else: return np.array(grad_W), np.array(grad_b)

def ComputeGradsNumSlow(X, Y, _lambda, W, b, gamma, beta, mean, var, batch_norm, h=0.000001):
    
    grad_W = [W_l.copy() for W_l in W]
    grad_b = [b_l.copy() for b_l in b]
    if batch_norm:
        grad_gamma = [gamma_l.copy() for gamma_l in gamma]
        grad_beta = [beta_l.copy() for beta_l in beta]
    
    c = ComputeCost(X, Y, _lambda, W, b, gamma, beta, batch_norm, mean, var)

    k = len(W)
    for l in range(k):
        
        for i in range(b[l].shape[0]):
            b_try = [b_l.copy() for b_l in b]
            b_try[l][i,0] += h
            c2 = ComputeCost(X, Y, _lambda, W, b_try, gamma, beta, batch_norm, mean, var)
            grad_b[l][i,0] = (c2-c)/h
        
        for i in range(W[l].shape[0]):
            for j in range(W[l].shape[1]):
                W_try = [W_l.copy() for W_l in W]
                W_try[l][i,j] += h
                c2 = ComputeCost(X, Y, _lambda, W_try, b, gamma, beta, batch_norm, mean, var)
                grad_W[l][i,j] = (c2-c)/h
                
        if l<(k-1) and batch_norm:
            
            for i in range(gamma[l].shape[0]):
                gamma_try = [gamma_l.copy() for gamma_l in gamma]
                gamma_try[l][i,0] += h
                c2 = ComputeCost(X, Y, _lambda, W, b, gamma_try, beta, batch_norm, mean, var)
                grad_gamma[l][i,0] = (c2-c)/h
            
            for i in range(beta[l].shape[0]):
                beta_try = [beta_l.copy() for beta_l in beta]
                beta_try[l][i,0] += h
                c2 = ComputeCost(X, Y, _lambda, W, b, gamma, beta_try, batch_norm, mean, var)
                grad_beta[l][i,0] = (c2-c)/h
    
    if batch_norm:
        return np.array(grad_W), np.array(grad_b), np.array(grad_gamma), np.array(grad_beta)
    else:
        return np.array(grad_W), np.array(grad_b)


def testing_gradients(X, Y, y, W, b, _lambda, h=1e-5, gamma=None, beta=None,batch_norm=False):

    if batch_norm:
        P_batch, X_batch, S, S_normalized, means, vars = EvaluateClassifier(X, W, b, gamma, beta, batch_norm)
        grad_W_anal, grad_b_anal, grad_gamma_anal, grad_beta_anal = ComputeGradients(X, Y, P_batch, X_batch, W, _lambda, means, vars,S=S,S_BN=S_normalized,gamma=gamma,beta=beta,batch_norm=batch_norm)
        grad_W_num, grad_b_num, grad_gamma_num, grad_beta_num = ComputeGradsNumSlow(X, Y, _lambda, W, b, gamma, beta, means, vars, batch_norm=batch_norm, h=0.000001)

    else:
        P_batch,X_batch = EvaluateClassifier(X, W, b)
        grad_W_anal, grad_b_anal = ComputeGradients(X, Y, P_batch, X_batch, W, _lambda)
        grad_W_num, grad_b_num = ComputeGradsNumSlow(X, Y, _lambda, W, b, gamma=None, beta=None, mean=None, var=None, batch_norm=batch_norm)

    epsilon = 1e-6

    grad_W_diff = np.abs(grad_W_anal-grad_W_num)
    grad_b_diff = np.abs(grad_b_anal-grad_b_num)
    if batch_norm:
        grad_gamma_diff = np.abs(grad_gamma_anal-grad_gamma_num)
        grad_beta_diff = np.abs(grad_beta_anal-grad_beta_num)

    error_w_num_slow_sum = [np.sum(diff>epsilon)*100 for diff in grad_W_diff]
    error_w_num_slow_acc = [np.mean(diff>epsilon)*100 for diff in grad_W_diff]
    error_b_num_slow_sum = [np.sum(diff>epsilon)*100 for diff in grad_b_diff]
    error_b_num_slow_acc = [np.mean(diff>epsilon)*100 for diff in grad_b_diff]
    if batch_norm:
        error_gamma_num_slow_sum = [np.sum(diff>epsilon)*100 for diff in grad_gamma_diff]
        error_gamma_num_slow_acc = [np.mean(diff>epsilon)*100 for diff in grad_gamma_diff]
        error_beta_num_slow_sum = [np.sum(diff>epsilon)*100 for diff in grad_beta_diff]
        error_beta_num_slow_acc = [np.mean(diff>epsilon)*100 for diff in grad_beta_diff]

    error_rel_w = [grad_W_diff[i] / np.maximum(epsilon, np.abs(grad_W_num[i])+np.abs(grad_W_anal[i])) for i in range(len(grad_W_diff))]
    error_rel_b = [grad_b_diff[i] / np.maximum(epsilon, np.abs(grad_b_num[i])+np.abs(grad_b_anal[i])) for i in range(len(grad_b_diff))]
    if batch_norm:
        error_rel_gamma = [grad_gamma_diff[i] / np.maximum(epsilon, np.abs(grad_gamma_num[i])+np.abs(grad_gamma_anal[i])) for i in range(len(grad_gamma_diff))]
        error_rel_beta = [grad_beta_diff[i] / np.maximum(epsilon, np.abs(grad_beta_num[i])+np.abs(grad_beta_anal[i])) for i in range(len(grad_beta_diff))]

    error_rel_w_sum = [np.sum(err>epsilon) for err in error_rel_w]
    error_rel_w_acc = [np.mean(err>epsilon)*100 for err in error_rel_w]
    error_rel_b_sum = [np.sum(err>epsilon) for err in error_rel_b]
    error_rel_b_acc = [np.mean(err>epsilon)*100 for err in error_rel_b]
    if batch_norm:
        error_rel_gamma_sum = [np.sum(err>epsilon) for err in error_rel_gamma]
        error_rel_gamma_acc = [np.mean(err>epsilon)*100 for err in error_rel_gamma]
        error_rel_beta_sum = [np.sum(err>epsilon) for err in error_rel_beta]
        error_rel_beta_acc = [np.mean(err>epsilon)*100 for err in error_rel_beta]


    print("Using error as number of points where difference is less than epsilon = 1e-6", end='\n\n')
    print(f"There are {error_w_num_slow_sum} errors for weights (error rate = {error_w_num_slow_acc} (in %))")
    print(f"There are {error_b_num_slow_sum} errors for biases (error rate = {error_b_num_slow_acc} (in %))")
    if batch_norm:
        print(f"There are {error_gamma_num_slow_sum} errors for gammas (error rate = {error_gamma_num_slow_acc} (in %))")
        print(f"There are {error_beta_num_slow_sum} errors for betas (error rate = {error_beta_num_slow_acc} (in %))")

    print()

    print("Using relative error as number of points where difference is less than epsilon = 1e-6", end='\n\n')
    print(f"There are {error_rel_w_sum} errors for weights (error rate = {error_rel_w_acc} (in %))")
    print(f"There are {error_rel_b_sum} errors for biases (error rate = {error_rel_b_acc} (in %))")
    if batch_norm:
        print(f"There are {error_rel_gamma_sum} errors for gammas (error rate = {error_rel_gamma_acc} (in %))")
        print(f"There are {error_rel_beta_sum} errors for betas (error rate = {error_rel_beta_acc} (in %))")

def MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma=None,beta=None,batch_norm=False,dropout_rate=0.0,add_noise=False, norm_before=True):
    n = train_X.shape[1]

    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']
    n_batch = GDparams['n_batch']
    n_cycles = GDparams['n_cycles']

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    # dict for nodes to drop per batch
    nodes_dropout = {}

    batches = []
    max_index = n//n_batch
    for j in range(max_index):
        j_start = j*n_batch
        j_end = (j+1)*n_batch
        inds = range(j_start, j_end)

        X_batch = train_X[:, inds]
        Y_batch = train_Y[:, inds]
        y_batch = [train_y[index] for index in inds]

        batches.append([X_batch,Y_batch,y_batch])
        
        nodes_dropout[j] = [None]*len(hidden_dimensions)
        for i in range(len(hidden_dimensions)):
            nodes_dropout[j][i] = np.random.rand(W[i].shape[0], n_batch) < dropout_rate

    batch_index = 0
    shuffle = True
    mean_av = None
    var_av = None
    alpha = 0.9
    for cycle in range(n_cycles):
        for t in range(2*cycle*n_s, 2*(cycle+1)*n_s):
            if (2*cycle*n_s <= t < (2*cycle+1)*n_s):
                eta = eta_min+(t-2*cycle*n_s)/n_s*(eta_max-eta_min)
            elif ((2*cycle+1)*n_s <= t < 2*(cycle+1)*n_s):
                eta = eta_max-(t-(2*cycle+1)*n_s)/n_s*(eta_max-eta_min)

            X, Y, y = batches[batch_index]
            if add_noise: X += np.random.normal(size=X.shape, loc=0, scale=0.1)
            
            if batch_norm: 
                P_batch, X_batch, S, S_normalized, means, vars = EvaluateClassifier(X, W, b, gamma, beta, batch_norm, norm_before=norm_before)
                for i in range(len(hidden_dimensions)): S_normalized[i] *= nodes_dropout[batch_index][i]
                grad_W, grad_b, grad_gamma, grad_beta = ComputeGradients(X, Y, P_batch, X_batch, W, _lambda, means, vars,S=S,S_BN=S_normalized,gamma=gamma,beta=beta,batch_norm=batch_norm)
                # exponential moving avg for means and variances with param alpha
                if not mean_av or not var_av: 
                    mean_av = means
                    var_av = vars
                else:
                    mean_av = [alpha*mean_av[l]+(1-alpha)*means[l] for l in range(len(means))]
                    var_av = [alpha*var_av[l]+(1-alpha)*vars[l] for l in range(len(vars))]
            else: 
                P_batch, X_batch = EvaluateClassifier(X, W, b)
                grad_W, grad_b = ComputeGradients(X, Y, P_batch, X_batch, W, _lambda)

            # weight updates
            for i in range(len(W)):
                W[i] += -eta*grad_W[i]
                b[i] += -eta*grad_b[i]
            if batch_norm:
                gamma = [gamma[l]-eta*grad_gamma[l] for l in range(len(gamma))]
                beta = [beta[l]-eta*grad_beta[l] for l in range(len(beta))]

            if shuffle:
                batch_index = np.random.randint(0,max_index)
            else:
                index+=1
                if index>=max_index:
                    index = 0

            # use means and vars
            if batch_norm:
                train_loss = ComputeCost(X, Y, _lambda, W, b, gamma, beta, batch_norm, mean_av, var_av)
                train_acc = ComputeAccuracy(X, y, W, b, gamma, beta, batch_norm, mean_av, var_av)
                val_loss = ComputeCost(val_X, val_Y, _lambda, W, b, gamma, beta, batch_norm, mean_av, var_av)
                val_acc = ComputeAccuracy(val_X, val_y, W, b, gamma, beta, batch_norm, mean_av, var_av)
            else:
                train_loss = ComputeCost(X, Y, _lambda, W, b)
                train_acc = ComputeAccuracy(X, y, W, b)
                val_loss = ComputeCost(val_X, val_Y, _lambda, W, b)
                val_acc = ComputeAccuracy(val_X, val_y, W, b)

            if t%10==0:
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

            sys.stdout.write(f"\rIter {t+1}/{2*n_s*n_cycles}: loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")
            sys.stdout.flush()

        #print(f"Cycle {cycle+1}: loss={train_loss} acc={train_acc} val_loss={val_loss} val_acc={val_acc}")
    print()

    if batch_norm: return W,b,gamma,beta,history
    else: return W, b, history


def plot_and_save_learning_curve(history, name, xlabel='update step', varying_lambda=False):

    train_acc = history['train_acc']
    val_acc = history['val_acc']
  
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # fig.suptitle('Plots for these paramerers')

    if not varying_lambda:
        xrange = np.array(range(len(train_acc)))*10
        ax1.plot(xrange, train_acc, label='Train')
        ax1.plot(xrange, val_acc, label='Validation')
        
        ax2.plot(xrange, train_loss, label='Train')
        ax2.plot(xrange, val_loss, label='Validation')
        
    else:
        ax1.plot(history['log lambda'],train_acc, label='Train')
        ax1.plot(history['log lambda'],val_acc, label='Validation')

        ax2.plot(history['log lambda'],train_loss, label='Train')
        ax2.plot(history['log lambda'],val_loss, label='Validation')

    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.legend()

    ax2.set_ylabel('Loss')
    ax2.set_xlabel(xlabel)
    ax2.legend()

    # plt.show()
    fig.tight_layout(pad=3.0)
    plt.savefig(name)

if __name__ == "__main__":

    # read the data
    train_X1, train_Y1, train_y1 = _LoadBatch('cifar-10-batches-py/data_batch_1')
    train_X2, train_Y2, train_y2 = _LoadBatch('cifar-10-batches-py/data_batch_2')
    train_X3, train_Y3, train_y3 = _LoadBatch('cifar-10-batches-py/data_batch_3')
    train_X4, train_Y4, train_y4 = _LoadBatch('cifar-10-batches-py/data_batch_4')
    train_X5, train_Y5, train_y5 = _LoadBatch('cifar-10-batches-py/data_batch_5')

    train_X = np.hstack((train_X1, train_X2, train_X3, train_X4, train_X5))
    train_Y = np.hstack((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5))
    train_y = np.array(train_y1+train_y2+train_y3+train_y4+train_y5)

    val_size = 1000

    indexes = np.random.choice(train_X.shape[1], val_size, replace=False)

    val_X = train_X[:, indexes]
    val_Y = train_Y[:, indexes]
    val_y = train_y[indexes]

    test_X, test_Y, test_y = _LoadBatch('cifar-10-batches-py/test_batch')
    
    train_X, val_X, test_X = normalize(train_X, val_X, test_X)

    batch_norm = False

    bonus_lambda_search = False
    bonus_bigger_network = False # done
    bonus_normalize_after = True # done
    bonus_dropout = False       # done
    bonus_add_noise = False      # done

    if bonus_lambda_search:
        print("Lambda")

        X = train_X
        Y = train_Y
        y = train_y

        input_dimension = X.shape[0]
        hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
        output_dimension = Y.shape[0]

        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

        print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")


        lambda_best = 0.0001
        # _lambda = 0.005
        _lambda = lambda_best
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':100,
            'n_cycles':2
            }
        # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
        GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True)
        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
        print(f"Test Accuracy: {test_acc}")
        name = f"{len(hidden_dimensions)+1} layers network with batch norm"
        plot_and_save_learning_curve(history, name)

    elif bonus_bigger_network:
        print("Bigger network")

        X = train_X
        Y = train_Y
        y = train_y

        input_dimension = X.shape[0]
        # hidden_dimensions = [50,100,150,100,50] # [50, 30, 20, 20, 10, 10, 10, 10] # 0.5323
        hidden_dimensions = [50,50,50,50,50] # 0.4997
        # hidden_dimensions = [100False,100,100,100,100] # 0.5224
        output_dimension = Y.shape[0]

        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

        print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")

        lambda_best = 0.0001
        # _lambda = 0.005
        _lambda = lambda_best
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':100,
            'n_cycles':2
            }
        # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
        GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True, dropout_rate=0.5)
        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
        print(f"Test Accuracy: {test_acc}")
        name = f"ex3_big_net_2.png"
        plot_and_save_learning_curve(history, name)

    elif bonus_normalize_after:
        print("Batch norm after nonlinear activation function")

        X = train_X
        Y = train_Y
        y = train_y

        input_dimension = X.shape[0]
        hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
        output_dimension = Y.shape[0]

        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

        print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")

        lambda_best = 0.0001
        # _lambda = 0.005
        _lambda = lambda_best
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':100,
            'n_cycles':2
            }
        # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
        GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True, norm_before=False)
        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
        print(f"Test Accuracy: {test_acc}") # before: 0.5111 , after: 0.4533
        name = f"ex3_batch_norm_after.png"
        plot_and_save_learning_curve(history, name)


    elif bonus_dropout:
        print("Dropout")

        X = train_X
        Y = train_Y
        y = train_y

        input_dimension = X.shape[0]
        hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
        output_dimension = Y.shape[0]

        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

        print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")


        lambda_best = 0.0001
        # _lambda = 0.005
        _lambda = lambda_best
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':100,
            'n_cycles':2
            }
        # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
        GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True, dropout_rate=0.5)
        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
        print(f"Test Accuracy: {test_acc}") # 0.3:0.5116, 0.5:0.512 , 0.7: 
        name = f"ex3_dropout.png"
        plot_and_save_learning_curve(history, name)

    elif bonus_add_noise:
        print("Add noise")

        X = train_X
        Y = train_Y
        y = train_y

        input_dimension = X.shape[0]
        hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
        output_dimension = Y.shape[0]

        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

        print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")

        lambda_best = 0.0001
        # _lambda = 0.005
        _lambda = lambda_best
        GDparams = {
            'eta_min':1e-5,
            'eta_max':1e-1,
            'n_batch':100,
            'n_cycles':2
            }
        # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
        GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True, add_noise=True)
        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
        print(f"Test Accuracy: {test_acc}") # 0.509
        name = f"ex3_noise.png"
        plot_and_save_learning_curve(history, name)


    elif not batch_norm:
        testing = False
        if testing:

            X = train_X[0:20,0:5]
            Y = train_Y[:,0:5]
            y = train_y[:5]
            _lambda = 0.0

            input_dimension = X.shape[0]
            hidden_dimensions = [50]
            output_dimension = Y.shape[0]

            W,b = initialize_weights(input_dimension, hidden_dimensions, output_dimension)

            testing_gradients(X, Y, y, W, b, _lambda, h=1e-5, batch_norm=False)
        
        else:
            sensitivity_analysis = True

            if sensitivity_analysis:

                print(f"Running without batch norm with 3 hidden layers for sensitivity analysis")

                X = train_X
                Y = train_Y
                y = train_y

                input_dimension = X.shape[0]
                hidden_dimensions = [50,50]
                output_dimension = Y.shape[0]

                lambda_best = 0.0001
                # _lambda = 0.005
                _lambda = lambda_best
                GDparams = {
                    'eta_min':1e-5,
                    'eta_max':1e-1,
                    'n_batch':100,
                    'n_cycles':2
                    }
                # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
                GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

                sigs = [1e-1,1e-3,1e-4]
                for sig in sigs:
                    W,b = initialize_weights(input_dimension, hidden_dimensions, output_dimension,He=False,sig=sig,batch_norm=False)
                    W,b,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda)
                    test_acc = ComputeAccuracy(test_X, test_y, W, b)
                    print(f"Test Accuracy for sigma={sig}: {test_acc}")
                    name = f"Sensitivy analysis for sigma={sig} with batch norm.png"
                    plot_and_save_learning_curve(history, name)

            else:
                # net1 net2 net3
                X = train_X
                Y = train_Y
                y = train_y

                input_dimension = X.shape[0]
                hidden_dimensions = [50, 30, 20, 20, 10, 10, 10, 10] # [50,50]
                output_dimension = Y.shape[0]

                W,b = initialize_weights(input_dimension, hidden_dimensions, output_dimension)

                _lambda = 0.005
                GDparams = {
                    'eta_min':1e-5,
                    'eta_max':1e-1,
                    'n_batch':100,
                    'n_cycles':2
                    }
                GDparams['n_s'] = int(5*45000/GDparams['n_batch'])

                W,b,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda)
                test_acc = ComputeAccuracy(test_X, test_y, W, b)
                print(f"Test Accuracy: {test_acc}")
                name = f"{len(hidden_dimensions)+1} layers network without batch norm"
                plot_and_save_learning_curve(history, name)

    else:
        testing = False

        if testing:
            X = train_X[0:20,0:10]
            Y = train_Y[:,0:10]
            y = train_y[:10]
            _lambda = 0

            input_dimension = X.shape[0]
            hidden_dimensions = [50,50]
            output_dimension = Y.shape[0]

            W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

            testing_gradients(X, Y, y, W, b, _lambda, h=1e-5, gamma=gamma, beta=beta,batch_norm=True)

        else:
            grid_search = False

            if grid_search:

                X = train_X
                Y = train_Y
                y = train_y

                input_dimension = X.shape[0]
                hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
                output_dimension = Y.shape[0]

                W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

                print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")


                l_max, l_min = -1, -5
                l = l_min+(l_max-l_min)*np.random.rand(10)
                lambdas_coarse = list(10**l)
                lambdas_coarse.sort()

                list_lambdas_fine = np.arange(0.0001, 0.0008, 0.00025)
                accuracies = {}
                for _lambda in list_lambdas_fine:
                    GDparams = {
                        'eta_min':1e-5,
                        'eta_max':1e-1,
                        'n_batch':100, 
                        'n_cycles':2
                        }
                    GDparams['n_s'] = int(5*45000/GDparams['n_batch'])

                    W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True)
                    test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
                    accuracies[_lambda] = test_acc
                print(accuracies)
            else:
                sensitivity_analysis = True
                if sensitivity_analysis:

                    print(f"Running batch norm with 3 hidden layers for sensitivity analysis")

                    X = train_X
                    Y = train_Y
                    y = train_y

                    input_dimension = X.shape[0]
                    hidden_dimensions = [50,50]
                    output_dimension = Y.shape[0]

                    lambda_best = 0.0001
                    # _lambda = 0.005
                    _lambda = lambda_best
                    GDparams = {
                        'eta_min':1e-5,
                        'eta_max':1e-1,
                        'n_batch':100,
                        'n_cycles':2
                        }
                    # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
                    GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

                    sigs = [1e-1,1e-3,1e-4]
                    for sig in sigs:
                        W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension,He=False,sig=sig,batch_norm=True)
                        W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True)
                        test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
                        print(f"Test Accuracy for sigma={sig}: {test_acc}")
                        name = f"Sensitivy analysis for sigma={sig} with batch norm.png"
                        plot_and_save_learning_curve(history, name)

                else:

                    X = train_X
                    Y = train_Y
                    y = train_y

                    input_dimension = X.shape[0]
                    hidden_dimensions = [50,50] # [50, 30, 20, 20, 10, 10, 10, 10]
                    output_dimension = Y.shape[0]

                    W,b,gamma,beta = initialize_weights(input_dimension, hidden_dimensions, output_dimension, batch_norm=True)

                    print(f"Running batch norm with {len(hidden_dimensions)} hidden layers")


                    lambda_best = 0.0001
                    # _lambda = 0.005
                    _lambda = lambda_best
                    GDparams = {
                        'eta_min':1e-5,
                        'eta_max':1e-1,
                        'n_batch':100,
                        'n_cycles':2
                        }
                    # GDparams['n_s'] = int(5*45000/GDparams['n_batch'])
                    GDparams['n_s'] = int(2*45000/GDparams['n_batch'])

                    W,b,gamma,beta,history = MiniBatchGD(train_X, train_Y, train_y, val_X, val_Y, val_y, GDparams, W, b, _lambda, gamma, beta , batch_norm=True)
                    test_acc = ComputeAccuracy(test_X, test_y, W, b, gamma, beta, batch_norm=True)
                    print(f"Test Accuracy: {test_acc}")
                    name = f"{len(hidden_dimensions)+1} layers network with batch norm"
                    plot_and_save_learning_curve(history, name)