
class RNN:
    def __init__(self):
        print("K")


if __name__=="__main__":
    fname = "Datasets/goblet_book.txt"
    data = open(fname, 'r').read()
    
    chars = set(data)
    
    char_to_ind = {}
    ind_to_char = {}
    for i, char in enumerate(chars):
        char_to_ind[char] = i
        ind_to_char[i] = char
        
    K = len(chars)
    m = 100
    rnn = RNN(input_size=K, hidden_size=m, output_size=K)
    
    print(rnn.b.shape, rnn.U.shape, rnn.W.shape, rnn.V.shape, rnn.c.shape)