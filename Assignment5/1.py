import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp


#1.1
#Defining the parameters

alpha= np.array([0.05,0.10,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.9,0.95,1])



#repeating 1000000 times
for i in range(20):
    variable =  np.random.binomial(1,0.05,size=1000000)
    print(variable)
sample =  np.random.binomial(20,0.05,size=1000000)
print(sample)



def simulate(bias, shape, seed) :
    scp.random.seed(seed)
    data = scp.random.binomial(1, bias, shape)
    return pd.DataFrame(data)


bias = 0.05
n = 20
N = 1000000

experiment = simulate(bias, (N, n), 2017)
experiment['mean'] = experiment.mean(axis = 1)

freqs = (experiment[[0,'mean']].groupby('mean').count() / N).reset_index().rename(columns = {0 : 'freq'})




#plotting the empirical frequency of sum of all values greater than alpha
prob1= sample/20
   
prob2=np.zeros(len(alpha))
for (i,a) in enumerate(alpha):
               prob2[i]=np.mean(prob1>=a)
               print(prob2[i],a)
               
                
               
            
plt.plot(alpha,prob2,"*-",label = 'frequency') 
plt.xlabel(r'$\alpha$ values ')
plt.ylabel('Probability')
plt.title('Empirical frequency')
plt.legend()
plt.show()
    

#hoeffdings bound              
def hoeffding_bound(bias, alpha, n) :
    return scp.exp(-2 * n * ((alpha - bias) ** 2))


freqs = freqs[freqs['mean']>=0.05]

freqs['hoef_bound'] = freqs['mean'].apply(lambda x : hoeffding_bound(bias, x, n))


"""Here we take the upper Markov’s bound
It is bias/(alpha) """

markov=[]

for i in alpha:
    markov.append(0.05/(i)) 
    
freqs['mark_bound'] = freqs['mean'].apply(lambda x : markov(bias, x))  


"""Here we take the upper Chebyshev’s bound """


def cheby():
    cheb=np.zeros(len(alpha))
    for i in range(len(alpha)):
        cheb[i]=0.002325/(alpha[i]-0.05)**2
    return cheb

array_c=cheby() 


sample_a=np.array([  1, 0.93, 0.2325, 0.10333333, 0.058125,
       0.0372, 0.02583333, 0.01897959, 0.01453125, 0.01148148,
       0.0093, 0.00768595, 0.00645833, 0.00550296, 0.0047449,
       0.00413333, 0.00363281, 0.00321799, 0.00287037, 0.00257618])
    
    
    
    
plt.plot(alpha,prob2,"*-",label = 'frequency') 
plt.plot(freqs['mean'], freqs['hoef_bound'],"*-")
plt.plot(alpha,markov,"o-") 
plt.plot(alpha,sample_a,"o-")
plt.legend(["Empirical frequency","hoeffdings bound","Markov's Bound","Chebyshev’s"])
plt.xlabel(r'$\alpha$ values ')
plt.ylabel('Probability')
plt.title('Plottings ')
plt.show()