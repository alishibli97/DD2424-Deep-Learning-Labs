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

        # # Clipping gradients
        # for parameter in ['b','c','U','W','V']:
        #     GRADS[parameter] = np.clip(GRADS[parameter], -5, 5)

        return GRADS
