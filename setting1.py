import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

list_patients=np.random.normal(loc=0,scale=1,size=(1000,3))

def run(triple): 
    #kiem dinh Ho: mu=0 vs H1: mu!=0
    # test-statistic 
    T=np.random.choice(triple)
    # two-sided p-value
    cdf=norm.cdf(T,loc=0,scale=1)# tich vo huong cua eta_T@sigma voi xichma=1 ma eta_T la[0,,1,0] nen sau khi nhan thi van ra scale=1
    p_value=2*min(cdf,1-cdf) 
    return p_value

if __name__=='__main__':
    list_p_value=[]
    for triple in list_patients:
        p_value=run(triple)
        list_p_value.append(p_value)
    plt.hist(list_p_value)
    plt.show()

        