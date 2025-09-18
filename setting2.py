import numpy as np
import gendata
import matplotlib.pyplot as plt
from scipy.stats import norm 

def run():
    true_beta = [0,0,0]
    X, y_obs = gendata.gen_data(true_beta)

    # Estimate beta
    XTX = np.dot(X.T, X)
    XTXinv = np.linalg.inv(XTX)
    XTXinvXT = np.dot(XTXinv, X.T)
    beta = np.dot(XTXinvXT, y_obs)

    # beta is considered as the test-statistic and can be decomposed in the form of eta^T y
    eta = XTXinvXT.T

    # Observed test-statistic
    etaTy_obs = np.dot(eta.T, y_obs)[0][0]  # This should be equal to beta

    # Compute two-sided naive-p value
    cdf = norm.cdf(etaTy_obs, loc=0, scale=np.sqrt(np.dot(eta.T, eta)[0][0]))
    naive_p_value = 2 * min(1 - cdf, cdf)

    return naive_p_value

if __name__ == '__main__':
    listp_value=[]
    for i in range(1000):
        p_value=run()
        listp_value.append(p_value)
    plt.hist(listp_value)
    plt.show()