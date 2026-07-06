import math

# List in Python

numbers = [10, 20, 30]

print(numbers) #output: [10, 20, 30]

print(numbers[0]) #output: 10

print(numbers[-1]) #output: 30

numbers.append(40) #output: [10, 20, 30, 40]

numbers.insert(1, 12) #output: [10, 12, 20, 30, 40]

print(numbers) #output: [10, 12, 20, 30, 40]

numbers.remove(12) #output: [10, 20, 30, 40]

print(numbers) #output: [10, 20, 30, 40]

numbers.pop(); #output: [10, 20, 30]

print(numbers) #output: [10, 20, 30]

numbers.pop(1) #output: [10, 30]

print(numbers) #output: [10, 30]

print(len(numbers)) #output: 2

# loop

for number in numbers:
    print(number, end=", ")
    
numbers = [11, 22, 33, 44]
print() # for next line
print(numbers) #output: [11, 22, 33, 44]

#slice 

print(numbers[:3]) #output: [11, 22, 33]

print(numbers[-2:]) #output: [33, 44]

print(numbers[1:4]) #output: [22, 33, 44]

#Tuple in Python

points = (10, 20, 30)

print(points) #output: (10, 20, 30)

# points[1] = 25 # This will raise an error because tuples are immutable

points = (10, 20)

x, y = points

print(x) #output: 10
print(y) #output: 20


# Dictionary in Python

student = {
    "name": "John Doe",
    "age": 20,
    "courses": ["Math", "Science", "History"]
}

print(student) #output: {'name': 'John Doe', 'age': 20, 'courses': ['Math', 'Science', 'History']}

print(student["name"]) #output: John Doe

student["age"] = 21 #output: {'name': 'John Doe', 'age': 21, 'courses': ['Math', 'Science', 'History']}

del student["courses"] #output: {'name': 'John Doe', 'age': 21}

print(student) #output: {'name': 'John Doe', 'age': 21}

for key, value in student.items():
    print(key, value) #output: name John Doe, age 21


# Set in Python

numbers = {1, 2, 3, 4, 5}

print(numbers) #output: {1, 2, 3, 4, 5}

numbers.add(6) #output: {1, 2, 3, 4, 5, 6}

numbers.remove(3) #output: {1, 2, 4, 5, 6}

numbers.discard(10) #output: {1, 2, 4, 5, 6} (no error if the element is not present)

numbers = {1,1,1,1,1,1}

print(numbers) #output: {1} (duplicates are removed in a set)

math_set = [1, 2, 3, 4, 5]

math.sqrt(16) #output: 4.0
math.ceil(4.2) #output: 5
math.floor(4.8) #output: 4