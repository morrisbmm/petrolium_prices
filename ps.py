#scatterplot describing relationship between gasoline and crude oil prices

import pandas as pd
import xlrd
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_excel('prices.xlsx')
print(df)

#cols = list(df.columns)
#w = df['year']
x = df['crude_oil_prices']
y = df['Gasoline_prices']

##########################using best_fit_slope_and_ intercept####################################################

'''def best_fit_slope_and_intercept(x,y):
    m = ((mean(x)*mean(y)) - mean(x*y)) / ((mean(x)*mean(x) - mean(x*x)))
    b= mean(y) - m*mean(x)
    return m,b
m,b = best_fit_slope_and_intercept(x,y)
print(m,b)
regression_line = [(m*x)+b for x in x]

def predict_crude_oil_price():
    a=input("enter the crude oil price here:")
    c = (m*int(a))+b
    plt.scatter(a,c,color="g")
    print(c)
    

plt.scatter(x,y,color='b')
plt.plot(x,regression_line,color='r')
#plt.scatter(w,y,label='Crude_oil_prices')
plt.title('RELATIONSHIP BETWEEN GASOLINE AND CRUDE OIL PRICES',color='b')
plt.ylabel('crude oil prices',color='b')
plt.xlabel('Gasoline prices',color='b')
#plt.legend()
plt.grid(True,color='k')
plt.show()

predict_crude_oil_price()'''


######################################## using batch gradient descent#################################################
X = x.to_numpy()
Y = y.to_numpy()
m = Y.size


#ones = np.array((m,1))
#X = np.hstack([X1,ones])

X = np.stack([np.ones(m), X], axis=1)


'''def cost(X,Y,theta):
    J = (1/(2*m))*np.sum((np.dot(X,theta)-y)**2)
    return J

costJ = cost(X,Y,theta = np.array([-1,2]))
print(costJ)

def gradient_descent(X,Y,theta,alpha,num_iters):
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        h = np.dot(X,theta)
        theta = theta - alpha*(1/m)*np.dot(X.transpose(),(h-y))
        J_history.append(cost(X,y,theta))

    return theta, J_history

theta = np.zeros(2)
iterations = 1500
alpha = 0.001

theta, J_history = gradient_descent(X ,y, theta, alpha, iterations)
print(theta)
print(J_history)'''


#########################################using the normal equations method######################################################
theta = np.linalg.inv(X.T@X)@X.T@y
print(theta)


def plotData(X,y):
    #fig = plt.figure()
    plt.plot(X[:,1], y, 'ro', ms=10, mec='k')
    plt.ylabel('Gasoline_prices $10,000')
    plt.xlabel('Crude_oil_prices in 10,000s')
    plt.plot(X[:, 1], np.dot(X, theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

plotData(X,y)


predicted = np.dot([1,200],theta)
print(predicted)


