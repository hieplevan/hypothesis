import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import gendata

def run(X,y_obs):
    #tinh normal equation
    XTX=np.dot(X.T,X)
    XTX_inv=np.linalg.inv(XTX)
    XTXinvXT=np.dot(XTX_inv,X.T)
    # Test statistic
    T=np.dot(XTXinvXT,y_obs)[0][0]
    #tinh eta_T 
    eta_T= XTXinvXT
    # Distribution of eta_T under H0
    cdf=norm.cdf(T,loc=0,scale= np.sqrt(np.dot(eta_T,eta_T.T)[0][0]))
    p_value=2*min(1-cdf,cdf)
    return p_value


if __name__ == '__main__':
    list_p_value=[]
    max_iteration = 1200
    for i in range(max_iteration):
        true_beta=[0,0,0]   
        listX,y_obs=gendata.gen_data(100,3,true_beta)
        listX2=[]
        for triple in listX:
            listX2.append([triple[1]])
        listX2=np.array(listX2)
        p_value=run(listX2,y_obs)
        list_p_value.append(p_value)
    plt.hist(list_p_value)
    plt.show()