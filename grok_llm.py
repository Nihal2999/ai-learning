from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # free, very capable
    temperature=1.0,
    messages=[
        {
            "role": "system",
            "content": "You always respond in valid JSON only. No extra text, no markdown."
        #    "content": "You are explaining to a 10-year-old. Use simple words and fun examples."
        #    "content": "You are a senior backend engineer. Be very technical, use exact terminology, no analogies."
        #    "content": "You are a concise assistant who explains things simply."
        },
        {
            "role": "user",
            "content": "Write a SQL query to find the top 5 customers by total order value from tables: customers(id, name) and orders(id, customer_id, amount)"
        #    "content": "Give me 3 facts about database indexes as a JSON array."
        #    "content": "What is a database index and why does it matter?"
        }
    ]
)

answer = response.choices[0].message.content
print(answer)

print(f"\n--- Token Usage ---")
print(f"Input tokens:  {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens:  {response.usage.total_tokens}")