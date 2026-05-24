# import anthropic
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# print(os.environ.get("ANTHROPIC_API_KEY"))
# client = anthropic.Anthropic()
#
# # Below line is sending request to Claude
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",  # the model we'll use
#     max_tokens=256,
#     messages=[
#         {"role": "user", "content": "Say hello in one sentence."}
#     ]
# )
#
# print(response.content[0].text)

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.choices[0].message.content)
