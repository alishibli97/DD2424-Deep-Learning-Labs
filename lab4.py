import numpy as np
import copy
from loguru import logger
import matplotlib.pyplot as plt

def one_hot_encoding(data, K, seq_length):
    # seq_length = len(data)
    result = np.zeros((K,seq_length))
    for i,d in enumerate(data):
        result[char_to_ind[d],i] = 1
    return result

class RNN:
    def __init__(self,m,eta,K,seq_length):

        self.m = m
        self.K = K
        self.eta = eta
        self.seq_length = seq_length

        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))

        sig = 0.1
        self.U = np.random.rand(m,K)*sig
        self.W = np.random.rand(m,m)*sig
        self.V = np.random.rand(K,m)*sig

    def tanh(self,x):
        x = np.clip(x, -700, 700)
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def softmax(self,x):
        x = np.clip(x, -700, 700)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def synthesize(self, h0, x0, n):

        Y = np.zeros((self.K,n))
        h_prev = h0
        x = x0
        for t in range(n):
            a = self.W@h_prev + self.U@x + self.b
            h = self.tanh(a)
            o = self.V@h + self.c
            p = self.softmax(o)

            # cp = p.cumsum(axis=0)
            # a = np.random.uniform(0,1)
            # ixs = (cp-a>0).nonzero()[0]
            # ii = ixs[0]
            # Y[ii,t] = 1

            x = np.random.multinomial(1, np.squeeze(p))[:,np.newaxis]
            Y[:,[t]] = x[:,[0]]

            h_prev = h

        return Y

    # def synthesize(self, x0, h0, length, stop_character_one_hot=None):
    #     Y = np.zeros(shape=(self.output_size,length))
    #     x = x0
    #     h_prev = h0
    #     for t in range(length):
    #         a = self.W@h_prev+self.U@x+self.b
    #         h = self.tanh(a)
    #         o = self.V@h+self.c
    #         p = self.softmax(o)

    #         # Create next sequence input randomly from predicted output distribution
    #         x = np.random.multinomial(1, np.squeeze(p))[:,np.newaxis]

    #         # Save the one-hot encoding of created next sequence input
    #         Y[:,[t]] = x[:,[0]]
            
    #         # Break loop if created next sequence input is equal to given stop character
    #         if all(x==stop_character_one_hot):
    #             Y = Y[:,0:(t+1)]
    #             break

    #         # Update previous hidden state for next sequence iteration
    #         h_prev = h

    #     return Y


    def ComputeLoss(self,Y,p):
        loss = 0
        for t in range(self.seq_length):
            loss -= np.log(Y[:,[t]].T@p[t])[0][0]
        return loss

    def forward(self,X,Y,h0):
        a = np.zeros((self.seq_length,self.m,1))
        h = np.zeros((self.seq_length,self.m,1))
        o = np.zeros((self.seq_length,self.K,1))
        p = np.zeros((self.seq_length,self.K,1))
        L = 0
        for t in range(self.seq_length):
            if t==0: a[t] = self.W@h0 + self.U@X[:,[t]] + self.b
            else: a[t] = self.W@h[t-1] + self.U@X[:,[t]] + self.b
            h[t] = self.tanh(a[t])
            o[t] = self.V@h[t] + self.c
            p[t] = self.softmax(o[t])

        L = self.ComputeLoss(Y,p)        
        h = np.insert(h,0,h0,axis=0)
        return a,h,p,L

    def backward(self, X, Y, p, h, a):

        grads = {
            "U": 0,
            "W": 0,
            "V": 0,
            "b": 0,
            "c": 0
        }

        h0 = h[0]
        h = h[1:]

        dL_do = []
        for t in range(self.seq_length):
            dL_do.append(-(Y[:,[t]]-p[t]).T)
            grads['V'] += dL_do[t].T@h[t].T

        dL_dhTao = dL_do[-1]@self.V
        dL_daTao = dL_dhTao@np.diag(1-self.tanh(a[-1][:,0])**2)
        
        dL_dh = np.zeros((self.seq_length,dL_dhTao.shape[0],dL_dhTao.shape[1])); dL_dh[-1] = dL_dhTao
        dL_da = np.zeros((self.seq_length,dL_daTao.shape[0],dL_daTao.shape[1])); dL_da[-1] = dL_daTao

        for t in range(self.seq_length-2,-1,-1):
            dL_dh[t] = dL_do[t]@self.V +dL_da[t+1]@self.W
            dL_da[t] = dL_dh[t]@np.diag(1-self.tanh(a[t])[:,0]**2)
        
        for t in range(self.seq_length):
            if t==0: grads['W'] += dL_da[t].T@h0.T
            else: grads['W'] += dL_da[t].T@h[t-1].T
            grads['U'] += dL_da[t].T@X[:,[t]].T

            grads['b'] += dL_da[t].T
            grads['c'] += dL_do[t].T

        # Clipping gradients
        for key in grads:
            grads[key] = np.clip(grads[key], -5, 5)

        return grads

    def run(self,book_data,max_updates):
        """
        You are now ready to write the high-level loop to train your RNN with the
        text in book data. The general high-level approach will be as follows. Let e
        (initialized to 1) be the integer that keeps track of where in the book you are.
        At each iteration of the SGD training you should grab a labelled training
        sequence of length seq length characters. Thus your sequence of input
        characters corresponds to book data(e:e+seq length-1) and the labels for
        this sequence is book data(e+1:e+seq length). You should convert these
        sequence of characters into the matrices X and Y (the one-hot encoding
        vectors of each character in the input and output sequences).
        """
        m_params = {
            'W': np.zeros_like(self.W),
            'U': np.zeros_like(self.U),
            'V': np.zeros_like(self.V),
            'b': np.zeros_like(self.b),
            'c': np.zeros_like(self.c)
        }

        h0 = np.zeros((m,1))

        smooth_loss = []
        epoch = 0
        update = 0
        while update < max_updates:
            hprev = h0
            epsilon = 1e-6
            for e in range(0, len(book_data)-1, self.seq_length):
                if e>len(book_data)-seq_length-1: 
                    epoch+=1
                    break
                if update>=max_updates: break

                # preparing data sequences
                X_chars = book_data[e:e+seq_length]
                Y_chars = book_data[e+1:e+seq_length+1]

                X = one_hot_encoding(X_chars,self.K,seq_length)
                Y = one_hot_encoding(Y_chars,self.K,seq_length)

                # forward and backward algorithm
                a,h,p,loss = rnn.forward(X, Y, hprev)
                grads = rnn.backward(X, Y, p, h, a)

                # AdaGrad updates
                m_params['W'] += grads['W']**2
                self.W -= (self.eta/np.sqrt(m_params['W']+epsilon))*grads['W']

                m_params['U'] += grads['U']**2
                self.U -= (self.eta/np.sqrt(m_params['U']+epsilon))*grads['U']

                m_params['V'] += grads['V']**2
                self.V -= (self.eta/np.sqrt(m_params['V']+epsilon))*grads['V']

                m_params['b'] += grads['b']**2
                self.b -= (self.eta/np.sqrt(m_params['b']+epsilon))*grads['b']

                m_params['c'] += grads['c']**2
                self.c -= (self.eta/np.sqrt(m_params['c']+epsilon))*grads['c']

                if not smooth_loss: smooth_loss.append(loss)
                else: smooth_loss.append(.999*smooth_loss[-1] + .001*loss)

                if update%10000==0:
                    message = f"Epoch={epoch} iter={update} loss={loss} smooth_loss={smooth_loss[-1]}"
                    logger.info(message)
                    Y_pred = self.synthesize(hprev,X[:,[0]],200)
                    result = []
                    for index in range(Y_pred.shape[1]):
                        character = ind_to_char[np.where(Y_pred[:,index]>0)[0][0]]
                        result.append(character)
                    result = "".join(result)
                    print(f"Predicted sequences: \n{result}\n")
                
                update+=1
                hprev = h[self.seq_length]

        return smooth_loss


def ComputeGradsNum(RNN, X, Y, h0, h=1e-4):
    grads = {}
    for param in ['V','W','U','b','c']:
        grads[param] = np.zeros_like(vars(RNN)[param])
        for i in range(vars(RNN)[param].shape[0]):
            for j in range(vars(RNN)[param].shape[1]):
                RNN_try = copy.deepcopy(RNN)
                vars(RNN_try)[param][i,j] += h
                _, _, _, loss2 = RNN_try.forward(X, Y, h0)
                vars(RNN_try)[param][i,j] -= 2*h
                _, _, _, loss1 = RNN_try.forward(X, Y, h0)
                grads[param][i,j] = (loss2-loss1)/(2*h)
    return grads

def testing_gradients(rnn,X,Y,h0):

    vars = ['V','W','U','b','c']

    a,h,p,L = rnn.forward(X,Y,h0)
    grads_anal = rnn.backward(X, Y, p, h, a)
    
    grads_num = ComputeGradsNum(rnn, X, Y, h0)

    epsilon = 1e-6

    errors ={
        var: {
            'error_max':0,
            'absolute_error_sum':0,
            'absolute_error_acc':0,
            'relative_error_sum':0,
            'relative_error_acc':0,
        }
        for var in vars
    }
    for var in vars:
        diffs = np.abs(grads_anal[var]-grads_num[var])

        errors[var]['error_max'] = np.max(diffs)

        errors[var]['absolute_error_sum'] = np.sum([np.sum(diff>epsilon) for diff in diffs])
        errors[var]['absolute_error_acc'] = np.mean([np.mean(diff>epsilon)*100 for diff in diffs])

        error_rel = [diffs[i] / np.maximum(epsilon, np.abs(grads_num[var][i])+np.abs(grads_anal[var][i])) for i in range(len(diffs))]
        errors[var]['relative_error_sum'] = np.sum([np.sum(err>epsilon) for err in error_rel])
        errors[var]['relative_error_acc'] = np.mean([np.mean(err>epsilon)*100 for err in error_rel])

    # print(errors)
    import pprint
    pprint.pprint(errors)

def plot_and_save_learning_curve(loss, name):

    _, ax = plt.subplots(1, 1, figsize=(15,5))
    plt.title('Learning curve '+name)
    ax.plot(range(1, len(smooth_loss)+1), smooth_loss)

    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")

    # plt.show()
    plt.savefig(name)


if __name__=="__main__":
    
    book_name = "Datasets/goblet_book.txt"
    book_data = open(book_name).read()

    K = len(set(book_data))
    
    char_to_ind = {}
    ind_to_char = {}
    for i,char in enumerate(set(book_data)):
        char_to_ind[char] = i
        ind_to_char[i] = char

    # params of the network: hidden layer dimension, learning rate, sequence length
    m = 5
    eta = 0.1
    seq_length = 25

    rnn = RNN(m,eta,K,seq_length)

    testing = True

    if testing:
        h0 = np.zeros((m,1))

        X_chars = book_data[0:seq_length]
        Y_chars = book_data[1:seq_length+1]

        X = one_hot_encoding(X_chars,K,seq_length)
        Y = one_hot_encoding(Y_chars,K,seq_length)

        # Y_pred = rnn.synthesize(h0,X[:,[0]],seq_length)
        # result = "".join([ind_to_char[i] for i in np.argmax(Y_pred,axis=0)])

        testing_gradients(rnn,X,Y,h0)

    else:
        max_updates = 100000
        smooth_loss = rnn.run(book_data,max_updates=max_updates)

        plot_and_save_learning_curve(smooth_loss,f"Smooth loss for {max_updates} steps - NEW")

        # passage of length 1000
        print()
        e=0
        seq_length=1000
        X_chars = book_data[e:e+seq_length]
        Y_chars = book_data[e+1:e+seq_length+1]
        X = one_hot_encoding(X_chars,K,seq_length)
        Y = one_hot_encoding(Y_chars,K,seq_length)
        h0 = np.zeros((m,1))
        Y_pred = rnn.synthesize(h0,X[:,[0]],1000)
        result = []
        for index in range(Y_pred.shape[1]):
            character = ind_to_char[np.where(Y_pred[:,index]>0)[0][0]]
            result.append(character)
        result = "".join(result)
        print(f"Predicted passage of length 1000: \n{result}\n")