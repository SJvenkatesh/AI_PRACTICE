#List comprehensions are a concise way to create lists in Python. They allow you to generate a new list by applying an expression to each item in an existing iterable (like a list or range) and optionally filtering items based on a condition.
squares = [i*i for i in range(10)]
print(squares) #output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

numbers = [1, 2, 3, 4, 5]

print([num*2 for num in numbers]) #output: [1, 2, 3, 4, 5]


numbers = [1, 2, 3, 4, 5]
even_numbers = [num for num in numbers if num % 2 == 0]
print(even_numbers) #output: [2, 4]


responses = ["python", "", "c++", "javascript"]

clean = [response for response in responses if response]
print(clean) #output: ['python', 'c++', 'javascript']

#Dictionary comprehensions are similar to list comprehensions, but they create dictionaries instead of lists. They allow you to generate a new dictionary by applying an expression to each item in an existing iterable and optionally filtering items based on a condition.
squares_dict = {i: i*i for i in range(10)}
print(squares_dict) #output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}

#set

numbers = [1, 2, 2, 2, 5]
unique_numbers = {num for num in numbers}
print(unique_numbers) #output: {1, 2, 5}

#enumerate

for index, value in enumerate(["a", "b", "c"]):
    print(index, value) #output: 0 a, 1 b, 2 c  

#zip

names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old.") #output: Alice is 25 years old., Bob is 30 years old., Charlie is 35 years old.

#map

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
print(squared_numbers) #output: [1, 4, 9, 16, 25]


#filter

numbers = [1, 2, 3, 4, 5]

even_numbers = list(filter(lambda x: x**2 == 0, numbers))
print(even_numbers) #output: [2, 4]

#iterators

numbers = [1, 2, 3, 4, 5]

it = iter(numbers)

print(next(it)) #output: 1
print(next(it)) #output: 2
print(next(it)) #output: 3

#generators : generators are a way to create iterators in Python. They allow you to generate values on the fly, rather than storing them all in memory at once. This can be useful for working with large datasets or infinite sequences.

def generate_numbers(n):
    for i in range(n):
        yield i  # The yield statement is used to produce a value and pause the function's execution, allowing it to be resumed later.
    

for number in generate_numbers(5):
    print(number) #output: 0, 1, 2, 3, 4

print(list(generate_numbers(5))) #output: [0, 1, 2, 3, 4]


def generate_numbers():
    yield 1
    yield 2
    yield 3

for number in generate_numbers():
    print(number) #output: 1, 2, 3

#genrator expressions are similar to list comprehensions, but they create generators instead of lists. They allow you to generate values on the fly, rather than storing them all in memory at once.

squared_numbers = (x**2 for x in range(10))

print(list(squared_numbers)) #output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

for number in squared_numbers:
    print(number) #output: (No output, because the generator has already been exhausted by the previous list() call)

#print(list(squared_numbers)) #output: [] (The generator has already been exhausted, so it returns an empty list)

#calling generator expression again to create a new generator we can do this by reassigning the generator expression to the variable squared_numbers again.
squared_numbers = (x**2 for x in range(10))

for number in squared_numbers:
    print(number) #output: 0, 1, 4, 9, 16, 25, 36, 49, 64, 81