import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

### UCB of mean via Hoeffding-Bentkus hybridization
def HB_mu_plus(muhat, n, delta, maxiters=1000):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n) 
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        try:
          return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)
        except:
          print(f"BRENTQ RUNTIME ERROR at muhat={muhat}") 
          return 1.0

def WSR_mu_plus(x, delta, maxiters=1000): # this one is different.
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1,n+1)))
    sigma2hat = (np.cumsum((x - muhat)**2) + 0.25) / (1 + np.array(range(1,n+1))) 
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log( 1 / delta ) / n / sigma2hat), 1)
    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)
    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1-1e-10, maxiter=maxiters)


if __name__ == "__main__":
    print(HB_mu_plus(0.1, 10000, 0.1, 1000))
    print(WSR_mu_plus(0.1+np.random.random(size=(1000,))/100, 0.01, 1000))
