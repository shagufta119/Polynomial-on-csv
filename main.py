import numpy
from sklearn import linear_model
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

speed = mymodel(17)
print(speed)



import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

model = numpy.poly1d(numpy.polyfit(x, y, 3))

line = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(line, model(line))
plt.show()



import pandas as pd
from sklearn import linear_model

df = pd.read_csv("CarPrice.csv")

X = df[['CarName', 'price']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
