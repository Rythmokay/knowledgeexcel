from google.generativeai import configure, GenerativeModel, list_models
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure with API key
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models (so you can check valid names)
print("Available models:")
for m in list_models():
    print(m.name, " — supports:", m.supported_generation_methods)

# Use a valid Gemini model name (replace with what you see above)
model = GenerativeModel("gemini-1.5-flash")  # or gemini-1.5-pro

# Generate content
response = model.generate_content("Hello from Gemini!")
print(response.text)
