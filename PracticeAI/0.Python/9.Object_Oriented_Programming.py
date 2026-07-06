
class Student:
    name = "John Doe"
    age = 20
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")


print(Student.name) #output: John Doe

student1 = Student("Alice", 22)
student1.greet() #output: Hello, my name is Alice and I am 22

print(student1.name) #output: Alice


class Human(Student):
    def __init__(self, name, age, gender):
        super().__init__(name, age)
        self.gender = gender
    def greet(self):
        print(f"Hello, my name is {self.name}, I am {self.age} years old and I am a {self.gender}.")


human1 = Human("Bob", 25, "male")
human1.greet() #output: Hello, my name is Bob, I am 25 years old and I am a male

#Composition

class Engine:
    def start(self):
        print("Engine started.")

class Car:
    def __init__(self):
        self.engine = Engine()  # Car has an Engine

    def start(self):
        self.engine.start()  # Delegating the start to the Engine


engine = Engine()
car = Car()
engine.start()  #output: Engine started.
car.start()     #output: Engine started.


#special methods

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
point1 = Point(2, 3)
point2 = Point(4, 5)

print(point1)  #output: Point(2, 3)
point3 = point1 + point2
print(point3)  #output: Point(6, 8)
print(point1 == point2)  #output: False 


