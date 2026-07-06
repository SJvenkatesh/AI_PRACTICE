name = "venkatesh"

def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet(name)) #output: Hello, venkatesh!

def add(a: int, b: int) -> int:
    return a+b

print(add(5, 3)) #output: 8

def process(items: list[str]) -> list[str]:
    return [item.upper() for item in items]

#list of strings

print(process(["apple", "banana", "cherry"])) #output: ['APPLE', 'BANANA', 'CHERRY']

#dictionary of strings, int

def process_dict(data: dict[str, int]) -> dict[str, int]:
    return {key: value*2 for key, value in data.items()}    

print(process_dict({"a": 1, "b": 2, "c": 3})) #output: {'a': 2, 'b': 4, 'c': 6}



#Dataclasses

from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    
person1 = Person("Alice", 30)
print(person1) #output: Person(name='Alice', age=30)

person1.age = 31
print(person1) #output: Person(name='Alice', age=31)

#person2 = Person("Bob") # This will raise an error because the age parameter is missing. To fix this, we can provide a default value for the age parameter in the Person class definition.

#print(person2) #output: Person(name='Bob', age=0)

#TypedDict : typedDict is a way to define a dictionary with specific key-value types in Python. It allows you to specify the expected types for the keys and values in a dictionary, providing better type checking and code clarity.

from typing import TypedDict 

class PersonDict(TypedDict):
    name: str
    age: int
    
person_dict: PersonDict = {"name": "Alice", "age": 30}
print(person_dict) #output: {'name': 'Alice', 'age': 30}

person_dict["age"] = 31
print(person_dict) #output: {'name': 'Alice', 'age': 31}

# person_dict2: PersonDict = {"name": "Bob", "age": 25, "gender": "male"} # This will raise an error because "gender" is not a valid key in the PersonDict TypedDict 
# print(person_dict2) #output: {'name': 'Bob'}


from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int


student1 = Student(name="Alice", age=20)
print(student1) #output: name='Alice' age=20

student1.age = 21
print(student1) #output: name='Alice' age=21

student2 = Student(name="Bob", age="25") # Pydantic will automatically convert the string "25" to an integer
print(student2) #output: name='Bob' age=25

#student3 = Student(name="Charlie", age="twenty") # This will raise a validation error because "twenty" cannot be converted to an integer
