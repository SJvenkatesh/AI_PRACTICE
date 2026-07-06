import requests

#get request to the API endpoint

response = requests.get(
    "https://jsonplaceholder.typicode.com/posts/1"
)

print(response.status_code)
print(response.json())

#post request to the API endpoint
data = {
    "name":"Venkatesh"
}
url = "https://jsonplaceholder.typicode.com/posts"
response = requests.post(url, json=data)
print(response.status_code)
print(response.json())

#OpenAI API request to the API endpoint

# from openai import OpenAI

# client = OpenAI()

# response = client.responses.create(
#     model="gpt-5.5",
#     input="Explain AI"
# )


