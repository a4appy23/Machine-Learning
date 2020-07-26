import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp
import scipy.stats as sts

def iterate(bias, shape, seed) :
    scp.random.seed(seed)
    data = scp.random.binomial(1, bias, shape)
    return pd.DataFrame(data)

               
#hoeffdings bound

def hoeffding(bias, alpha, n) :
    return scp.exp(-2 * n * ((alpha - bias) ** 2))






# adding the other plots from assi1 
#------------------------------------------------------------------------------
#2.4
"""Here we take the upper Markov’s bound
It is 1/(2*alpha) """


markov=[]

for i in alpha:
    markov.append(1/(2*i))

plt.plot(alpha,prob2,"o-")      
plt.plot(alpha,markov,"o-")        
plt.xlabel(r'$\alpha$ values ')
plt.ylabel('Probability')
plt.title('Empirical frequency with markov bound')
plt.show()


"""Here we take the upper Chebyshev’s bound """


def cheby():
    cheb=np.zeros(len(alpha))
    for i in range(len(alpha)):
        cheb[i]=1/(80*(alpha[i]-0.5)**2)
    return cheb

array_c=cheby() 
#replacing the values greater than 1 with trivial bound of 1.
sample_a=np.array([1,1,1,0.55555556, 0.3125,     0.2,
 0.13888889, 0.10204082, 0.078125,   0.0617284 , 0.05  ])


    
    
    
   
#1
#Defining the parameters

alpha= np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95,1])
sample =  np.random.binomial(20,0.5,size=1000000)
print(sample)
#plotting the empirical frequency of sum of all values greater than alpha
prob1= sample/20
   
prob2=np.zeros(len(alpha))
for (i,a) in enumerate(alpha):
               prob2[i]=np.mean(prob1>=a)
               print(prob2[i],a)
  



#simulating and plotting hoeffding bound
bias = 0.5
n = 20
N = 1000000

experiment = iterate(bias, (N, n), 1000)
experiment['mean'] = experiment.mean(axis = 1)

freqs = (experiment[[0,'mean']].groupby('mean').count() / N).reset_index().rename(columns = {0 : 'freq'})

freqs = freqs[freqs['mean']>=0.5]
freqs['hoef_bound'] = freqs['mean'].apply(lambda x : hoeffding(bias, x, n))


    
#calculating hoeffding bound
#-----------------------------------------------------------
#Prob that alpha =1:
print((0.5) ** 20, hoeffding(1,0.5,20))




#Prob that alpha equals =0.95:
p=sts.binom(20, 0.5).pmf(1) + sts.binom(20, 0.5).pmf(0)
print("--------------------probability------------")
print(p)
print("---------------------------the hoeffding bound=-------------")
print( hoeffding(0.95,0.5,20))



    
    
 
    
    
   #plotting the results              
plt.plot(alpha,prob2,"*-",label = 'frequency') 
plt.plot(freqs['mean'], freqs['hoef_bound'],"*-", label = 'hoeffding bound')             
plt.xlabel(r'$\alpha$ values ')
plt.ylabel('Probability')
plt.title('Empirical frequency')
plt.legend()
plt.show()  
    
 #all the plots together   

plt.plot(alpha,prob2,"o-")
plt.plot(alpha,markov,"o-")
plt.plot(alpha,sample_a,"o-")
plt.plot(freqs['mean'], freqs['hoef_bound'],"o-")
plt.ylim((0,1.25))
plt.legend(["Empirical frequency","Markov's Bound","Chebyshev’s",'hoeffding bound'])
plt.xlabel(r'$\alpha$ values ')
plt.ylabel('Probability')
plt.title('Plottings ')
plt.show()












