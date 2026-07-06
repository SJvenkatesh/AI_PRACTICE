import asyncio
import requests


async def hello():
    print("Hello")

asyncio.run(hello()) 

async def some_api():
    response = requests.get(
        "https://jsonplaceholder.typicode.com/posts/1"
    )
    print(response.status_code)
    print(response.json())

async def fetch():
    await some_api()

asyncio.run(fetch())