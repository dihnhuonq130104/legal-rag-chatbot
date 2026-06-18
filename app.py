import streamlit as st
import time
from generator import generate_answer

# Cấu hình trang
st.set_page_config(
    page_title="Trợ lý AI - Luật Lao Động Việt Nam",
    page_icon="⚖️",
    layout="centered"
)

# CSS (Giữ nguyên của bạn vì bạn làm rất đẹp)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafd 0%, #eef3f8 100%); }
    .title-card {
        text-align: center; padding: 24px; margin-bottom: 30px;
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px);
        border-radius: 18px; box-shadow: 0 10px 30px rgba(30, 61, 89, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.9);
    }
    .main-title { color: #1e3d59; font-weight: 700; font-size: 2.1rem; margin-bottom: 8px; }
    .subtitle { color: #17b978; font-size: 1.05rem; font-weight: 600; text-transform: uppercase; }
    .stChatMessage { border-radius: 16px !important; margin-bottom: 16px; padding: 18px !important; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #e3f2fd !important; border: 1px solid #cce5ff !important; }
    .stChatMessage[data-testid="stChatMessageAssistant"] { background-color: #ffffff !important; border: 1px solid #f0f0f0 !important; }
    .source-item-header { font-weight: 700; color: #1e3d59; font-size: 0.95rem; margin-top: 10px; border-bottom: 1px dashed #e0e0e0; padding-bottom: 4px; }
    blockquote { background-color: #f7f9fa !important; border-left: 4px solid #17b978 !important; padding: 12px 18px !important; margin: 6px 0 16px 0 !important; color: #495057 !important; font-size: 0.9rem !important; border-radius: 0 8px 8px 0; }
    [data-testid="stSidebar"] { background-color: #1e3d59 !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .sidebar-desc { color: #a0b2c6 !important; font-size: 0.88rem; line-height: 1.4; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-card">
        <div class="main-title">⚖️ Trợ lý AI Luật Lao Động</div>
        <div class="subtitle">Hệ thống Tra cứu chéo Luật & Nghị định hướng dẫn</div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Hệ sinh thái dữ liệu")
    st.markdown("""
    <p class="sidebar-desc">
    Hệ thống Agentic RAG đang hoạt động trên cơ sở dữ liệu pháp luật lao động:
    <br>• <b>Bộ luật Lao động 2019</b>
    <br>• <b>Nghị định 145 năm 2020</b>
    <br>• <b>Nghị định 152 năm 2020</b>
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 💡 Gợi ý câu hỏi:")
    st.markdown("""
    - *Thời gian thử việc tối đa của người quản lý doanh nghiệp là bao lâu?*
    - *Hồ sơ đề nghị cấp giấy phép lao động cho người nước ngoài gồm các giấy tờ nào?*
    - *Người sử dụng lao động có quyền áp dụng hình thức xử lý kỷ luật lao động nào?*
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Xin chào! Tôi là Trợ lý AI chuyên môn về Luật Lao động Việt Nam. Bạn cần trợ giúp thông tin nào?", 
            "sources": []
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Căn cứ pháp lý trích xuất"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"<div class='source-item-header'>🏷️ [{src['doc_type']}] {src['source_file']} | {src['dieu']}: {src['ten_dieu']}</div>", unsafe_allow_html=True)
                    st.markdown(f"> {src['text']}")

# Hàm giả lập chữ chạy (Simulated Streaming)
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) # Tốc độ chạy chữ

if user_query := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu hệ thống & đối chiếu Nghị định..."):
            # Chạy RAG ngầm
            result = generate_answer(user_query, chat_history=st.session_state.messages[:-1])
            answer = result["answer"]
            sources = result["sources"]
            
        # Khi có kết quả, hiển thị chữ chạy
        st.write_stream(stream_text(answer))
        
        if sources:
            with st.expander("📚 Căn cứ pháp lý trích xuất"):
                for idx, src in enumerate(sources):
                    st.markdown(f"<div class='source-item-header'>🏷️ [{src['doc_type']}] {src['source_file']} | {src['dieu']}: {src['ten_dieu']}</div>", unsafe_allow_html=True)
                    st.markdown(f"> {src['text']}")
                    
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })