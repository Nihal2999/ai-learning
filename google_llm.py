from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Only this changes — base_url + api_key
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    temperature=0.3,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is a database index and why does it matter?"}
    ]
)

print(response.choices[0].message.content)