import numpy as np
import copy
from loguru import logger
import matplotlib.pyplot as plt



def one_hot(character_index, number_distinct_characters):
    character_one_hot = np.zeros(shape=(number_distinct_characters,1))
    character_one_hot[character_index,0] = 1
    
    return character_one_hot

class RNN(object):
    
    def __init__(self, input_size, hidden_size, output_size, scale=0.01, seed=1):
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Initialize the parameters matrix
        print(hidden_size)
        self.b = np.zeros((hidden_size,1))
        self.c = np.zeros((output_size,1))
        np.random.seed(seed)
        self.U = np.random.normal(size=(hidden_size,input_size), loc=0, scale=scale)
        self.W = np.random.normal(size=(hidden_size,hidden_size), loc=0, scale=scale)
        self.V = np.random.normal(size=(output_size,hidden_size), loc=0, scale=scale)

    def tanh(self, a):
        a = np.clip(a, -700, 700)
        
        return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    
    def softmax(self, o):
        o = np.clip(o, -700, 700)
        exponential_o = np.exp(o)
        p = exponential_o/exponential_o.sum(axis=0)
        
        return p

    def synthesize(self, x0, h0, length, stop_character_one_hot=None):
        Y = np.zeros(shape=(self.output_size,length))
        x = x0
        h_prev = h0
        for t in range(length):
            a = self.W@h_prev+self.U@x+self.b
            h = self.tanh(a)
            o = self.V@h+self.c
            p = self.softmax(o)

            # Create next sequence input randomly from predicted output distribution
            x = np.random.multinomial(1, np.squeeze(p))[:,np.newaxis]

            # Save the one-hot encoding of created next sequence input
            Y[:,[t]] = x[:,[0]]
            
            # Break loop if created next sequence input is equal to given stop character
            if all(x==stop_character_one_hot):
                Y = Y[:,0:(t+1)]
                break

            # Update previous hidden state for next sequence iteration
            h_prev = h

        return Y

    def forward(self, X, Y, h0):
        
        # Create empty lists for storing the final and intermediary vectors (by sequence iterations)
        seq_length = X.shape[1]
        p, o, h, a = [None]*seq_length, [None]*seq_length, [None]*seq_length, [None]*seq_length
        
        # Iterate the input sequence of one hot encoded characters
        loss = 0
        for t in range(seq_length):
            if t==0:
                a[t] = self.W@h0+self.U@X[:,[t]]+self.b
            else:
                a[t] = self.W@h[t-1]+self.U@X[:,[t]]+self.b
            h[t] = self.tanh(a[t])
            o[t] = self.V@h[t]+self.c
            p[t] = self.softmax(o[t])
            loss -= np.log(Y[:,[t]].T@p[t])[0,0]

        return loss, p, [h0]+h, a

    def backward(self, X, Y, p, h, a):
        
        # Extract initial hidden state (sequence time 0)
        h0 = h[0]
        h = h[1:]
        
        # Initialize the gradients matrix
        GRADS = dict()
        for parameter in ['b','c','W','U','V']:
            GRADS[parameter] = np.zeros_like(vars(self)[parameter])
        
        # Iterate inversively the input sequence of one hot encoded characters
        seq_length = X.shape[1]
        grad_a = [None]*seq_length
        for t in range((seq_length-1), -1, -1):
            g = -(Y[:,[t]]-p[t]).T
            GRADS['V'] += g.T@h[t].T
            GRADS['c'] += g.T
            if t<(seq_length-1):
                dL_h = g@self.V+grad_a[t+1]@self.W
            else:
                dL_h = g@self.V
            grad_a[t] = dL_h@np.diag(1-h[t][:,0]**2)
            if t==0:
                GRADS['W'] += grad_a[t].T@h0.T
            else:
                GRADS['W'] += grad_a[t].T@h[t-1].T
            GRADS['U'] += grad_a[t].T@X[:,[t]].T
            GRADS['b'] += grad_a[t].T

        # Clipping gradients
        for parameter in ['b','c','U','W','V']:
            GRADS[parameter] = np.clip(GRADS[parameter], -5, 5)

        return GRADS
    
    def train(self,
              text,
              ind_to_char,
              char_to_ind,
              seq_length,
              eta=0.1,
              number_updates=100000,
              max_epochs=np.inf,
              find_best_network=True,
              continue_previous_training=False,
              verbose=False,
              verbose_show_loss_frequency=1000,
              verbose_show_sample_frequency=10000,
              verbose_show_sample_length=200,
              verbose_show_sample_stop_character=None):
        
        # Check if any current update or smooth loss exist in the RNN class
        previous_update_or_loss_exist = True
        try:
            self.current_update
            self.smooth_loss
        except:
            previous_update_or_loss_exist = False
        
        # If not continuation of previous training...
        if not continue_previous_training or not previous_update_or_loss_exist:
            
            # Initialize current update and smooth loss list
            self.current_update = 1
            self.smooth_loss = []
            
            # Create AdaGrad memory parameters matrix
            for parameter in ['b','c','U','W','V']:
                vars(self)[parameter+'_memory'] = np.zeros_like(vars(self)[parameter])
            
            # Define minimum loss and initialize best network parameters matrix (if required)
            if find_best_network:
                self.smooth_loss_min = np.inf
                self.smooth_loss_min_update = 1
                for parameter in ['b','c','U','W','V']:
                    vars(self)[parameter+'_best'] = vars(self)[parameter]

        # Iterate updates
        current_epoch = 1
        while self.current_update<=number_updates:
            
            # Define the initial previous hidden state for next epoch (full text iteration)
            h_prev = np.zeros(shape=(self.hidden_size,1))

            # Iterate input text by blocks of seq_length characters
            for e in range(0, len(text)-1, seq_length):
                
                if e>len(text)-seq_length-1:
                    break

                # Generate the sequence data for the iteration (one hot encoding for each character)
                X_chars, Y_chars = text[e:(e+seq_length)], text[(e+1):(e+1+seq_length)]
                X = np.zeros(shape=(self.input_size,seq_length))
                Y = np.zeros(shape=(self.output_size,seq_length))
                for t in range(seq_length):
                    X[:,[t]] = one_hot(char_to_ind[X_chars[t]], self.output_size)
                    Y[:,[t]] = one_hot(char_to_ind[Y_chars[t]], self.output_size)

                # Forward and backward pass
                loss, p, h, a = self.forward(X, Y, h_prev)
                newGRADS = self.backward(X, Y, p, h, a)

                # Store smoothed loss
                if self.current_update==1:
                    self.smooth_loss.append(loss)
                else:
                    self.smooth_loss.append(0.999*self.smooth_loss[-1]+0.001*loss)

                # AdaGrad update step
                for parameter in ['b','c','U','W','V']:
                    vars(self)[parameter+'_memory'] += newGRADS[parameter]**2
                    vars(self)[parameter] += -eta*newGRADS[parameter]/ \
                        np.sqrt(vars(self)[parameter+'_memory']+np.spacing(1))
                
                # If best loss improved this network iteration parameters (if required)
                if find_best_network and self.smooth_loss[-1]<self.smooth_loss_min:
                    self.smooth_loss_min = self.smooth_loss[-1]
                    self.smooth_loss_min_update = self.current_update
                
                if verbose:
                    
                    # Show loss
                    shown_loss = False
                    if self.current_update%verbose_show_loss_frequency==0 or self.current_update==1:
                        shown_loss = True
                        print('Update '+str(self.current_update)+' with loss: '+ \
                              str(self.smooth_loss[-1]))
                        
                    # Show a synthesized sample
                    if self.current_update%verbose_show_sample_frequency==0 or self.current_update==1:
                        synthesize_one_hot = \
                            self.synthesize(x0=X[:,[0]], h0=h_prev, length=verbose_show_sample_length,
                                            stop_character_one_hot=verbose_show_sample_stop_character)
                        synthesize_characters = []
                        for index in range(synthesize_one_hot.shape[1]):
                            character = ind_to_char[np.where(synthesize_one_hot[:,index]>0)[0][0]]
                            synthesize_characters.append(character)
                        if shown_loss:
                            print('Synthesized sample:\n'+''.join(synthesize_characters)+'\n')
                        else:
                            print('Update '+str(self.current_update)+' with loss: '+ \
                                  str(self.smooth_loss[-1])+'\nSynthesized sample:\n'+ \
                                  ''.join(synthesize_characters)+'\n')
                    
                self.current_update += 1
                if self.current_update>number_updates:
                    break

                # Update the previous hidden state for next iteration
                h_prev = h[seq_length]

            current_epoch += 1
            if current_epoch>max_epochs:
                break
        
        # Update the final training parameters with the best stored network (if required)
        if find_best_network:
            for parameter in ['b','c','U','W','V']:
                vars(self)[parameter] = vars(self)[parameter+'_best']


def ComputeGradsNum(RNN, X, Y, h0, h=1e-4):
   
    # Iterate parameters and compute gradients numerically
    GRADS = dict()
    for parameter in ['b','c','U','W','V']:
        GRADS[parameter] = np.zeros_like(vars(RNN)[parameter])
        for i in range(vars(RNN)[parameter].shape[0]):
            for j in range(vars(RNN)[parameter].shape[1]):
                RNN_try = copy.deepcopy(RNN)
                vars(RNN_try)[parameter][i,j] += h
                loss2, _, _, _ = RNN_try.forward(X, Y, h0)
                vars(RNN_try)[parameter][i,j] -= 2*h
                loss1, _, _, _ = RNN_try.forward(X, Y, h0)
                GRADS[parameter][i,j] = (loss2-loss1)/(2*h)
    
    return GRADS


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

    print(errors)

def plot_and_save_learning_curve(loss, name):

    _, ax = plt.subplots(1, 1, figsize=(15,5))
    plt.title('Learning curve '+name)
    ax.plot(range(1, len(smooth_loss)+1), smooth_loss)

    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")

    # plt.show()
    plt.savefig(name)

def plot_and_save_learning_curve(loss, name):

    _, ax = plt.subplots(1, 1, figsize=(15,5))
    plt.title('Learning curve '+name)
    ax.plot(range(1, len(loss)+1), loss)

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

    # rnn = RNN(m,eta,K,seq_length)

    testing = False

    if testing:
        h0 = np.zeros((m,1))

        X_chars = book_data[0:seq_length]
        Y_chars = book_data[1:seq_length+1]

        X = one_hot_encoding(X_chars,K,seq_length)
        Y = one_hot_encoding(Y_chars,K,seq_length)

        # Y_pred = rnn.synthesis(h0,X[:,[0]],seq_length)
        # result = "".join([ind_to_char[i] for i in np.argmax(Y_pred,axis=0)])

        testing_gradients(rnn,X,Y,h0)

    else:
        # max_updates = 100000
        # smooth_loss = rnn.run(book_data,max_updates=max_updates)

        book_chars = set(book_data)

        K = len(book_chars)
        m = 100
        myRNN = RNN(input_size=K, hidden_size=m, output_size=K)
        myRNN.b.shape, myRNN.U.shape, myRNN.W.shape, myRNN.V.shape, myRNN.c.shape

        seq_length = 25
        X_chars = book_data[0:seq_length]
        Y_chars = book_data[1:(1+seq_length)]
        X_chars, Y_chars

        X = np.zeros(shape=(K,seq_length))
        Y = np.zeros(shape=(K,seq_length))
        for t in range(seq_length):
            X[:,[t]] = one_hot(character_index=char_to_ind[X_chars[t]], number_distinct_characters=K)
            Y[:,[t]] = one_hot(character_index=char_to_ind[Y_chars[t]], number_distinct_characters=K)

        h0 = np.zeros(shape=(myRNN.hidden_size,1))
        loss, p, h, a = myRNN.forward(X, Y, h0)
        newGRADS = myRNN.backward(X, Y, p, h, a)



        myRNN.train(text=book_data,
                    ind_to_char=ind_to_char,
                    char_to_ind=char_to_ind,
                    seq_length=25,
                    eta=0.1,
                    number_updates=100000,
                    max_epochs=np.inf,
                    find_best_network=True,
                    continue_previous_training=False,
                    verbose=True,
                    verbose_show_loss_frequency=10000,
                    verbose_show_sample_frequency=10000,
                    verbose_show_sample_length=200,
                    verbose_show_sample_stop_character=None)

        plot_and_save_learning_curve(myRNN.smooth_loss,"Smooth loss for 100000 steps")

        synthesize_one_hot = myRNN.synthesize(x0=X[:,[0]], h0=h0, length=1000)
        synthesize_characters = []
        for index in range(synthesize_one_hot.shape[1]):
            character = ind_to_char[np.where(synthesize_one_hot[:,index]>0)[0][0]]
            synthesize_characters.append(character)
        print(''.join(synthesize_characters))
        