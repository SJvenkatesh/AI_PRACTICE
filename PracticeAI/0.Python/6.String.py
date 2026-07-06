# Creating a string

name = "Venkatesh"

print(name) #output: Venkatesh

# Multi-line string

multi_line_string = """This is a multi-line string.
It can span multiple lines.
You can use triple quotes to create multi-line strings."""

print(multi_line_string)

#output: This is a multi-line string.
#It can span multiple lines.
#You can use triple quotes to create multi-line strings.

# Indexing strings
print(name[1]) #output: e
print(name[-1]) #output: h

# Slicing strings
print(name[0:4]) #output: Venk
print(name[2:4]) #output: nk

# String immutability
# Strings are immutable, meaning you cannot change their content after they are created.

# name[0] = "v" # This will raise an error because strings are immutable

name = "Hello" # You can create a new string instead

print(name) #output: Hello

name = "J"+name[1:] # This creates a new string by concatenating "J" with the substring of name starting from index 1

print(name) #output: Jello

print(name.lower()) #output: jello
print(name.upper()) #output: JELLO

text = "   Hello, World!   "
print(text.strip()) #output: Hello, World! (removes leading and trailing whitespace)

text = "I like Java"

print(text.replace("Java", "Python")) #output: I like Python (replaces "Java" with "Python")

# split

sentance = "This is a sample sentence."
words = sentance.split() # Splits the sentence into a list of words based on whitespace
print(words) #output: ['This', 'is', 'a', 'sample', 'sentence.']

# join

words = ["This", "is", "a", "sample", "sentence."]
joined_sentence = " ".join(words) # Joins the list of words into a single string
print(joined_sentence) #output: This is a sample sentence.

# startswith and endswith

text = "Hello, World!"
print(text.startswith("Hello")) #output: True (checks if the string starts with "Hello")
print(text.endswith("World!")) #output: True (checks if the string ends with "World!")

# find

text = "Hello, World!"
print(text.find("World")) #output: 7 (returns the index of the first occurrence of "World" in the string, or -1 if not found)   

count = text.count("o") # Counts the number of occurrences of "o" in the string
print(count) #output: 2

# f-strings (formatted string literals)
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.") #output: My name is Alice and I am 30 years old. (uses f-strings to format

# Escape characters

text = "This is a line.\nThis is another line." # \n is an escape character for a new line
print(text) #output: This is a line.
# This is another line.

text = "He said, \"Hello!\"" # \" is an escape character for a double quote
print(text) #output: He said, "Hello!"


# string iteration

word = "Python"
for letter in word:
    print(letter, end=", ") #output: P, y, t, h, o, n,  (iterates through each letter in the string and prints it with a comma and space)

# string membership

text = "Artificial Intelligence"
print("Intelligence" in text) #output: True (checks if "Intelligence" is a substring of text)
print("Machine" not in text) #output: True (checks
