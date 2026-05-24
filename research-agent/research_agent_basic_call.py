from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# .env file should have: GROQ_API_KEY=gsk_your_key_here
client = Groq()

SYSTEM_PROMPT = """You are an expert research assistant.
Your job is to research a topic thoroughly and return a
well-structured, factual summary.
Be concise, clear, and cite what you know confidently."""

def research(topic: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # best free model on Groq
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Research this topic and give me a summary: {topic}"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    topic = "Impact of AI on software development in 2026"
    print(f"Researching: {topic}\n")
    print("=" * 50)
    print(research(topic))
