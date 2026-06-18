import os
import json
import sys
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Đảm bảo in unicode ra console không bị lỗi trên Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Kiểm tra xem GOOGLE_API_KEY có tồn tại không
if not os.getenv("GOOGLE_API_KEY"):
    print("CẢNH BÁO: Không tìm thấy biến môi trường GOOGLE_API_KEY trong file .env!")

# Import các thư viện LangChain và AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder

from retriever import LegalRetriever

# Khởi tạo retriever toàn cục
retriever = LegalRetriever()

# Khởi tạo mô hình Reranker CrossEncoder
print("Đang khởi tạo CrossEncoder Reranker 'BAAI/bge-reranker-v2-m3'...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("Khởi tạo Reranker thành công.")

# Danh sách lưu trữ các nguồn tài liệu đã được Agent sử dụng
accessed_sources = []

@tool
def lookup_specific_document(keyword: str, doc_type: str) -> str:
    """Tìm kiếm thông tin bổ sung trong một loại văn bản pháp luật cụ thể ('Luật' hoặc 'Nghị định').
    Hãy sử dụng công cụ này khi Context hiện tại chưa đủ thông tin, cần làm rõ chi tiết một từ khóa,
    hoặc khi nội dung Luật ghi 'Chính phủ quy định chi tiết' và bạn cần tra cứu thêm Nghị định hướng dẫn.
    
    Args:
        keyword: Từ khóa hay cụm từ cần tìm kiếm.
        doc_type: Loại văn bản pháp luật cần tìm kiếm, bắt buộc phải là 'Luật' hoặc 'Nghị định'.
    """
    global accessed_sources
    
    # Chuẩn hóa doc_type đầu vào
    doc_type_clean = doc_type.strip()
    if "luật" in doc_type_clean.lower():
        doc_type_clean = "Luật"
    elif "nghị định" in doc_type_clean.lower():
        doc_type_clean = "Nghị định"
    else:
        doc_type_clean = "Nghị định"
        
    print(f"\n[Tool Call] Đang tìm kiếm bổ sung từ khóa '{keyword}' trong '{doc_type_clean}'...")
    results = retriever.search(keyword, top_k=10, doc_type_filter=doc_type_clean)
    
    # Lưu lại metadata
    for r in results:
        meta = dict(r["metadata"])
        meta["text"] = r["text"]
        if not any(x["source_file"] == meta["source_file"] and x["dieu"] == meta["dieu"] and x.get("text") == meta["text"] for x in accessed_sources):
            accessed_sources.append(meta)
            
    if not results:
        return f"Không tìm thấy tài liệu bổ sung nào khớp với từ khóa '{keyword}' trong nhóm '{doc_type_clean}'."
        
    # Tạo chuỗi phản hồi cho Agent
    formatted = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        formatted.append(
            f"[Tài liệu bổ sung {i+1}]\n"
            f"File: {meta['source_file']} ({meta['doc_type']})\n"
            f"Vị trí: {meta['chuong']} -> {meta['dieu']}: {meta['ten_dieu']}\n"
            f"Nội dung: {r['text']}"
        )
    return "\n\n---\n\n".join(formatted)

# Khởi tạo LLM chuẩn
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# Cấu hình System Prompt
system_prompt = (
    "Bạn là Chuyên gia Luật Lao động Việt Nam. Dựa vào Context được cung cấp, hãy trả lời câu hỏi.\n\n"
    "Quy tắc BẮT BUỘC:\n"
    "1. Nếu Luật ghi \"Chính phủ quy định chi tiết\", BẮT BUỘC dùng tool 'lookup_specific_document' với doc_type='Nghị định' để tra cứu văn bản hướng dẫn.\n"
    "2. Mọi thông tin trả lời BẮT BUỘC có nguồn đi kèm dạng: (Khoản X, Điều Y - Tên văn bản).\n"
    "3. Trả lời chính xác, mạch lạc, format rõ ràng."
)

# Khởi tạo Agent chuẩn của Langchain
tools = [lookup_specific_document]
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def generate_answer(user_query, chat_history=None):
    """Hàm chính thực hiện RAG."""
    global accessed_sources
    accessed_sources = []
    
    try:
        # 1. Tra cứu top 8
        print(f"\n[RAG Pipeline] Tìm kiếm truy vấn: '{user_query}'...")
        retrieved = retriever.search(user_query, top_k=8)
        
        # 2. Reranking
        top_5 = []
        if retrieved:
            pairs = [[user_query, item["text"]] for item in retrieved]
            scores = reranker.predict(pairs)
            for j, item in enumerate(retrieved):
                item["rerank_score"] = float(scores[j])
            
            sorted_items = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)
            top_5 = sorted_items[:5]
        
        for item in top_5:
            meta = dict(item["metadata"])
            meta["text"] = item["text"]
            if not any(x["source_file"] == meta["source_file"] and x["dieu"] == meta["dieu"] and x.get("text") == meta["text"] for x in accessed_sources):
                accessed_sources.append(meta)
                
        # 3. Tạo Context
        context_str = ""
        for idx, item in enumerate(top_5):
            meta = item["metadata"]
            context_str += (
                f"Tài liệu {idx+1}: {meta['source_file']} ({meta['doc_type']})\n"
                f"Vị trí: {meta['chuong']} -> {meta['dieu']}: {meta['ten_dieu']}\n"
                f"Nội dung: {item['text']}\n---\n"
            )
            
        # Chuyển đổi chat_history từ Streamlit sang LangChain Messages
        langchain_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

        # 4. Chạy Agent
        print("[RAG Pipeline] Đang gọi Agent...")
        inputs = {
            "input": f"Context ban đầu:\n{context_str}\n\nCâu hỏi của người dùng: {user_query}",
            "chat_history": langchain_history
        }
        
        response = agent_executor.invoke(inputs)
        answer = response["output"]
        
        # Chuẩn hóa answer thành chuỗi nếu là danh sách hoặc đối tượng khác
        if isinstance(answer, list):
            extracted_texts = []
            for part in answer:
                if isinstance(part, dict) and "text" in part:
                    extracted_texts.append(part["text"])
                elif hasattr(part, "text"):
                    extracted_texts.append(part.text)
                elif isinstance(part, str):
                    extracted_texts.append(part)
                else:
                    extracted_texts.append(str(part))
            answer = "".join(extracted_texts)
        elif not isinstance(answer, str):
            answer = str(answer)
            
        return {
            "answer": answer,
            "sources": accessed_sources
        }
        
    except Exception as e:
        print(f"[RAG Pipeline] Lỗi: {e}")
        return {
            "answer": f"Đã xảy ra lỗi hệ thống: {str(e)}",
            "sources": []
        }