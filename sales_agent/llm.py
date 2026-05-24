# llm.py
# ============================================================
# LLM CONFIGURATION
# One place to configure the Groq LLM.
# Import this wherever you need the LLM.
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

load_dotenv()


def get_llm():
    """
    Returns a configured Groq LLM instance.
    Temperature 0.1 = focused and consistent output.
    Right for financial analysis — not creative writing.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=SecretStr(os.getenv("GROQ_API_KEY") or "")
    )

