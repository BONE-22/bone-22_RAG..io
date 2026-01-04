from langchain_community.document_loaders import PyPDFLoader #讀取檔案(pdf)
from langchain_community.vectorstores import FAISS #向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings #嵌入模型
from langchain_text_splitters import RecursiveCharacterTextSplitter #切分文字
from langchain_google_genai import ChatGoogleGenerativeAI #Google提供Embedding模型
from langchain_core.prompts import ChatPromptTemplate #提示詞模板
from langchain_core.output_parsers import StrOutputParser #輸出解析器
from configparser import ConfigParser #讀取與管理設定檔

# 1. 讀取配置
config = ConfigParser()
config.read("config.ini")

# 2. 載入單一 PDF (假設你已將兩者內容合併，或只測一個)
loader = PyPDFLoader("貿特198診斷報告final.pdf") # 替換為你的檔案名
data = loader.load()

# 3. 切分文本 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(data)

# 4. 向量化
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.from_documents(docs, embeddings)

# 5. 設定 LLM 
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=config["Gemini"]["API_KEY"]
)

# 6. 優化 Prompt
prompt = ChatPromptTemplate.from_template(
    """
    你是一個專業的導讀助手。請根據以下提供的上下文資訊來回答問題。
    如果資訊不足以回答，請誠實告知。
    
    回答要求：
    1. 使用繁體中文。
    2. 除了回覆答案外，請列出比較的清單（條列式，不要表格）。

    上下文：
    {context}

    問題：{input}
    """
)

# 7. 執行檢索與生成
query = "空壓系統做怎麼樣的改善"
docs_related = db.similarity_search(query, k=4)

# 將檢索到的 Document 物件轉換為純文字
context_combined = "\n---\n".join([doc.page_content for doc in docs_related])

chain = prompt | llm_gemini | StrOutputParser()
result = chain.invoke({
    "input": query,
    "context": context_combined
})

print("Question:", query)
print("LLM Answer:\n", result)
