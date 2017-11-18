from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Create_DataSet(time_series,p):
    
    Xtrain,Ytrain = np.array([[]]), np.array([[]])

    for i in range(len(time_series)-p):
        if Xtrain.any():
            Xtrain = np.append(Xtrain,np.array([time_series[i:i+p]]),axis=0)
            Ytrain = np.append(Ytrain,np.array([[time_series[i+p]]]),axis=0)
        else:
            Xtrain = np.array([time_series[i:i+p]])
            Ytrain = np.array([[time_series[i+p]]])
    return Xtrain,Ytrain    

file = open('haberman.data', 'r') 
data = file.read()
data = data.replace('\n',',')
data = data[:-1]
# print data
file_test = open('haberman.teste.data', 'r') 
data_test = file_test.read()
data_test = data_test.replace('\n',',')
data_test = data_test[:-1]
# print data

train =map(int,data.split(','))
# print train
test = map(int, data_test.split(','))

p = 4
Xtrain,Ytrain = Create_DataSet(train,p)
# print Xtrain, Ytrain

#Create neural network 
mlp = MLPClassifier(hidden_layer_sizes=(10),
    max_iter=1000,
    alpha = 1e-6, 
    solver='sgd',
    verbose=10,
    tol=1e-5,
    random_state=1,
    activation = "logistic", 
    learning_rate_init=.1)

#training neural network
mlp.fit(Xtrain, Ytrain)
    
Xtest,Ytest = Create_DataSet(test,p)
print("Taxa de acerto para " + str(p) + " entradas: " + str(mlp.score(Xtest,Ytest)))

Ypredict = mlp.predict(Xtest)

print "EQM para " + str(p) + " entradas:" +str(met.mean_squared_error(Ytest,Ypredict))

plt.figure(1)
plt.plot(np.arange(len(Ytest)),Ytest,'rs',label="desejado".format(1))
plt.plot(np.arange(len(Ytest)),Ypredict,'bs',label="previsto".format(2))
plt.legend()
plt.xlabel("Amostras")
plt.ylabel("Valor estimado")

plt.show()