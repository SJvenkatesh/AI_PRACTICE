import os
from dotenv import load_dotenv

load_dotenv() # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")

print(api_key)