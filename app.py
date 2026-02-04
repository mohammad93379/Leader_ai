import re
import json
import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# -----------------------
# مدل و اتصال
# -----------------------
MODEL = "qwen3:latest"
model = ChatOllama(model=MODEL)

# -----------------------
# بارگذاری داده‌ها از JSON جدید
# -----------------------
with open("data/Leaders_data.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)

docs = []
for person_data in data_json.get("all_data", []):
    person = person_data.get("person", "")
    for entry in person_data.get("datas", []):
        content = (
            f"{entry.get('Incident','')}\n"
            f"{entry.get('Conditions','')}\n"
            f"{entry.get('Decision','')}"
        )
        docs.append(Document(page_content=content, metadata={"person": person}))

# -----------------------
# ایجاد بردارها برای جستجوی semantical
# -----------------------
embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = DocArrayInMemorySearch.from_documents(documents=docs, embedding=embeddings)

try:
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
except Exception:
    retriever = None

# -----------------------
# تابع کمکی برای دریافت اسناد مرتبط
# -----------------------
def fetch_docs_for_query(retriever, vector_store, query, k=10):
    if retriever and hasattr(retriever, "get_relevant_documents"):
        try:
            return retriever.get_relevant_documents(query)
        except Exception:
            pass
    if retriever and hasattr(retriever, "retrieve"):
        try:
            return retriever.retrieve(query)
        except Exception:
            pass
    if vector_store:
        if hasattr(vector_store, "similarity_search"):
            try:
                return vector_store.similarity_search(query, k=k)
            except Exception:
                pass
        if hasattr(vector_store, "similarity_search_with_score"):
            try:
                pairs = vector_store.similarity_search_with_score(query, k=k)
                return [doc for doc, score in pairs]
            except Exception:
                pass
        if hasattr(vector_store, "search"):
            try:
                return vector_store.search(query, k=k)
            except Exception:
                pass
    return []

# -----------------------
# پرامپت اصلی برای پاسخ به سوال کاربر
# -----------------------
main_template = """
You are a helpful assistant. 

Rule 1: If the user query matches or is similar to the 'Incident' or 'Conditions' in the context, return the corresponding 'Decision'.
Rule 2: If the query is casual (e.g., "سلام"), respond briefly and friendly in Persian.
Rule 3: If no match is found, try to mix 'Incident' and 'Conditions' to make up some 'Decision' to replay.

Context:
{context}

User Query: {question}

Respond strictly following the rules.
"""
main_prompt = PromptTemplate.from_template(main_template)
main_chain = main_prompt | model | StrOutputParser()

# -----------------------
# توابع کمکی
# -----------------------
def clean_answer(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

def normalize_query(q: str) -> str:
    return re.sub(r"[^\u0600-\u06FF\s]", "", q or "").strip()

# -----------------------
# استریم‌لیت و استایل
# -----------------------
st.set_page_config(page_title="📘 Incident Decision Bot", page_icon="🤖", layout="centered")
st.markdown("""
<style>
.stApp, .css-1aumxhk, .stTextInput>div>div>input { direction: rtl; text-align: right; }
.stTitle h1 { direction: rtl; text-align: right; }
.stSubheader h3 { direction: rtl; text-align: right; }
.user-msg { background-color: #DCF8C6; padding: 10px 15px; border-radius: 15px 15px 0px 15px; float: right; max-width: 80%; clear: both; }
.bot-msg { background-color: #ECECEC; padding: 10px 15px; border-radius: 15px 15px 15px 0px; float: left; max-width: 80%; clear: both; }
.timestamp { font-size: 0.7em; color: gray; display: block; margin-top: 2px; }
.clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)

st.title("📘 چت‌بات تصمیمات رهبران")

# -----------------------
# وضعیت چت
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("موضوع یا شرایط رخداد را وارد کنید.")

if user_input:
    normalized = normalize_query(user_input)
    with st.spinner("در حال پردازش..."):
        docs_for_query = fetch_docs_for_query(retriever, vector_store, normalized, k=10)
        context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs_for_query]) if docs_for_query else ""
        try:
            raw_answer = main_chain.invoke({"context": context_text, "question": user_input})
            answer = clean_answer(raw_answer) or "پاسخی تولید نشد."
        except Exception as e:
            answer = f"خطا در تولید پاسخ از مدل: {e}"

        st.session_state.chat_history.append({
            "user": user_input,
            "bot": answer,
            "time": datetime.now().strftime("%H:%M:%S")
        })

# نمایش چت
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="user-msg">{chat["user"]}<span class="timestamp">{chat["time"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-msg">{chat["bot"]}<span class="timestamp">{chat["time"]}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
