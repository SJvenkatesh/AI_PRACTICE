numbers = [1, 2, 3]

#Numpy  stands for Numerical Python (High-performance array library)
import numpy as np
numbers = np.array([1,2,3])

print(numbers) #output : [10 20 30]

print(numbers.shape)

#printing matrix

matrix = np.array([[1, 2],[3,4]])

print(matrix)

#muliply matrix
matrix = matrix * 2
print(matrix)

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)

scores = np.array([90,80,100])
print(scores.mean()) #90.0

print(scores.max()) #100

print(scores.min()) #80


#Pandas is like Excel inside Python

#DataFrame
#The most important object.

import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Marks": [90, 85]
})

df = pd.read_csv("students.csv")

print("------------------First row----------------")
#First row
print(df.head())

print("---------------------Last row--------------")
#Last row
print(df.tail())

print("-----------------print columns------------------")
#Columns
print(df["Name"])

print("------------------Muliple columns---------------")

print(df[["Name","Marks"]])

#Filtering

print(df[df["Marks"]>85])

print("----------Adding Columns----------------")
df["Passed"] = True

print(df.head())

print("----------------------Average--------------------------")
print(df["Marks"].mean())

print("---------------Maximum-------------")
print(df["Marks"].max())



print("---------------Minimum-------------")
print(df["Marks"].min())

print(df.sort_values(by="Marks", ascending=False))

