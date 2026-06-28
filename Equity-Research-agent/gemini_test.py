import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv("local.env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in local.env")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.5-flash")

response = model.generate_content("tell me about corgi insurance")

print(response.text)
