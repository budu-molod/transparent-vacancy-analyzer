import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://opendata.trudvsem.ru/api/v1/vacancies/region/39")
MAX_VACANCIES = int(os.getenv("MAX_VACANCIES", "20"))
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "10"))
SCHEDULE_INTERVAL_MINUTES = int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "60"))
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output/report.json")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://models.github.ai/inference")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")