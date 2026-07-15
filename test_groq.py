"""
Quick Groq connectivity test 
"""
import os
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GROQ_API_KEY")
print("Key loaded:", "yes" if key else "NO — check your .env file")
if key:
    print("Key preview:", key[:6] + "..." + key[-4:])

try:
    from groq import Groq
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    print("SUCCESS ->", resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED -> {type(e).__name__}: {e}")
    print("\nIf this is a connection / SSL / timeout error, your network is blocking")
    print("api.groq.com. Try a mobile hotspot, Google Colab, or set your corporate proxy.")
