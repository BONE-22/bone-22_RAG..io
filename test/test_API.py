from langchain_google_genai import ChatGoogleGenerativeAI
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=config["Gemini"]["API_KEY"],
    max_retries=5
)


messages = [
    ("system", "你是一個哲學家，請用繁體中文回答。"),
    ("human", "人生的意義是什麼？"),
]

try:
    print("正在請求 Google API...")
    response = llm_gemini.invoke(messages)
    print(f"回答：\n{response.content}")
except Exception as e:
    print(f"連線失敗，請檢查 API Key 或模型名稱。錯誤訊息：\n{e}")
