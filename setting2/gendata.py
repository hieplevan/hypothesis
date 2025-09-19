import numpy as np

def gen_data(n,p,true_beta):

    # tao ra bo cac X gom 3 yeu to 
    X=np.random.normal(loc=0,scale=1,size=(n,p))
    true_beta=np.reshape(true_beta ,(p,1))

    #tao ra y=X1*beta1+X2*beta2+X3*beta3+noise
    true_y=np.dot(X,true_beta)
    noise=np.random.normal(loc=0,scale=1,size=(n,1))
    y=true_y+noise
    return X,y
