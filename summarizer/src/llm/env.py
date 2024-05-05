import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_CONTEXT_LENGTH = os.getenv("MODEL_CONTEXT_LENGTH")
