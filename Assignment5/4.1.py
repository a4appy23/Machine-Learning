import pandas as pd


#read the data
X_train = pd.read_csv('X_train.csv', header = None, sep = ' ')
y_train = pd.read_csv('y_train.csv', header = None, sep = ' ')

X_test = pd.read_csv('X_test.csv', header = None, sep = ' ')
y_test = pd.read_csv('y_test.csv', header = None, sep = ' ')

#data processing

df1=pd.DataFrame(X_train)
df2=pd.DataFrame(y_train)

df1['df2']=df2
df1
#total no of data points
data_full=33724
#frequency of label 0
label0=df1[(df1.df2 == 0)]
len(label0)
freq0=len(label0)/33724
#frequency of label 1
label1=df1[(df1.df2 == 1)]
len(label1)
freq1=len(label1)/33724
#frequency of label 2
label2=df1[(df1.df2 == 2)]
len(label2)
freq2=len(label2)/33724
#frequency of label 3
label3=df1[(df1.df2 == 3)]
freq3=len(label3)/33724
len(label3)

#frequency of label 4
label4=df1[(df1.df2 == 4)]
len(label4)
freq4=len(label4)/33724


print(freq0)
print(freq1)
print(freq2)
print(freq3)
print(freq4)


