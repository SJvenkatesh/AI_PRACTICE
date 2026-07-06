file = open("notes.txt", 'r')

content = file.read()

print(content)

file.close()

with open("notes.txt", 'w') as file:
    file.write("This is a new note.\n")
    file.write("This is another note.\n")

file = open("notes.txt", 'r')
content = file.read()
print(content)

with open("notes.txt", 'a') as file:
    file.write("\nThis is an appended note.\n") 

file = open("notes.txt", 'r')
content = file.read()
print(content)

# Reading JSON data

import json

with open("student.json", 'r') as file:
    student = json.load(file)
print(student)


# Writing JSON data

student = {
    "name": "John Doe",
    "age": 20,
    "courses": ["Math", "Science", "History"]
}

with open("student.json", 'w') as file:
    json.dump(student, file, indent=4) #indent=4 is used to format the JSON data with indentation for better readability

with open("student.json", 'r') as file:
    student = json.load(file)
print(student)


with open("student1.json", "w") as file:
    json_text = json.dumps(student) #dumps is used to convert the dictionary to a JSON string
    print(json_text)
    file.write(json_text)

with open("student1.json", "r") as file:
    student = json.load(file)

print("JSON data written to student1.json", student)