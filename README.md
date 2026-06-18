# ⚖️ Trợ lý AI - Tra Cứu Chéo Hệ Sinh Thái Luật Lao Động Việt Nam

Dự án RAG (Retrieval-Augmented Generation) thông minh sử dụng kiến trúc Agentic phối hợp với cơ sở dữ liệu Vector Qdrant local, mô hình nhúng đa ngôn ngữ, bộ định tuyến Rerank CrossEncoder và LLM Google Gemini để hỗ trợ tra cứu, lập luận và đối chiếu chéo giữa các văn bản Luật Lao động và Nghị định hướng dẫn thi hành.

---

## ✨ Điểm Nổi Bật Của Hệ Thống

1. **Context Enrichment (Làm giàu ngữ cảnh):** Parser tự động trích xuất cấu trúc văn bản pháp luật, ghép tiêu đề Điều luật vào từng Khoản nhỏ để tối ưu hóa độ liên quan ngữ nghĩa trong quá trình tìm kiếm vector.
2. **Local Vector Database (Qdrant & FastEmbed):** Lưu trữ vector cục bộ dưới máy với mô hình nhúng đa ngôn ngữ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. **Semantic Reranking:** Sử dụng CrossEncoder Reranker (`BAAI/bge-reranker-v2-m3`) chấm điểm lại top 15 kết quả và lọc lấy top 5 kết quả tốt nhất trước khi gửi tới LLM.
4. **Agentic Cross-Lookup Tool:** Tích hợp LangChain Agent với khả năng tự động gọi công cụ tìm kiếm bổ sung tài liệu (ví dụ: tự động tra cứu Nghị định đi kèm khi Luật Lao động có ghi *"Chính phủ quy định chi tiết"*).
5. **Giao diện Chat Premium:** Streamlit UI mượt mà, hỗ trợ lịch sử chat trực quan, hiển thị chi tiết nguồn căn cứ pháp lý và nội dung trích đoạn dưới dạng blockquote tiện lợi.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
├── data/                    # Thư mục chứa 3 file văn bản gốc (.docx)
│   ├── Bộ luật Lao động 2019.docx
│   ├── Nghị định 145 năm 2020.docx
│   └── Nghị định 152 năm 2020.docx
├── qdrant_db/               # Thư mục lưu trữ database Qdrant cục bộ (sau khi ingest)
├── data_parser.py           # Phân tích cấu trúc file .docx thành legal_corpus.json
├── retriever.py             # Lớp kết nối Qdrant DB và thực hiện Ingest dữ liệu + Tìm kiếm ngữ nghĩa
├── generator.py             # Pipeline Rerank, thiết lập LangChain Agent và gọi Gemini LLM
├── app.py                   # Giao diện chat trực quan bằng Streamlit
├── requirements.txt         # Khai báo các thư viện cần thiết
├── .env                     # Lưu trữ khóa GOOGLE_API_KEY (không push lên github)
└── README.md                # Hướng dẫn sử dụng dự án
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### 1. Khởi tạo Virtual Environment và Cài đặt Thư viện
Mở terminal tại thư mục dự án và thực hiện các lệnh sau:

**Trên Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Trên Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` nằm ở thư mục gốc của dự án với nội dung sau:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Hãy thay `your_gemini_api_key_here` bằng Google AI Studio API Key của bạn).*

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Trích xuất và nạp dữ liệu vào Vector DB
Nếu bạn muốn nạp dữ liệu từ đầu (hoặc sau khi cập nhật file `.docx` mới trong thư mục `data/`):

1. **Chạy Parser** để tạo file cấu trúc JSON:
   ```bash
   python data_parser.py
   ```
   *Kết quả tạo ra file `legal_corpus.json` chứa toàn bộ 1,623 chunks văn bản.*

2. **Ingest dữ liệu** vào Qdrant DB:
   Chạy trực tiếp file `retriever.py` để tự động hóa quy trình tạo Collection và nạp Vector:
   ```bash
   python retriever.py
   ```
   *Quá trình này sẽ sử dụng FastEmbed để nhúng tự động và lưu trữ vào thư mục `./qdrant_db`.*

### Bước 2: Khởi động giao diện ứng dụng
Chạy ứng dụng web bằng Streamlit:
```bash
streamlit run app.py
```
Hệ thống sẽ khởi động cổng dịch vụ. Bạn có thể truy cập qua trình duyệt tại:
👉 **http://localhost:8501**

---

## 💡 Luồng Hoạt Động Hệ Thống (RAG Pipeline)

1. **User Query:** Người dùng đặt câu hỏi trên Streamlit UI.
2. **Dense Retrieval:** Hệ thống sử dụng Qdrant tìm kiếm 15 đoạn văn bản liên quan nhất từ kho dữ liệu Luật/Nghị định.
3. **Reranker (CrossEncoder):** Sử dụng mô hình `BAAI/bge-reranker-v2-m3` chấm điểm lại độ tương đồng ngữ nghĩa thực tế của 15 đoạn trên với câu hỏi, lọc ra 5 đoạn điểm cao nhất.
4. **Agentic Reasoning (LangChain Agent & Gemini):**
   - Agent nhận context ban đầu (top 5).
   - Nếu phát hiện điều khoản Luật ghi *"Chính phủ quy định chi tiết..."*, Agent sẽ tự động kích hoạt công cụ `lookup_specific_document` để tra cứu chéo sang văn bản Nghị định tương ứng.
   - Tổng hợp thông tin và định dạng câu trả lời kèm nguồn tham chiếu bắt buộc dạng `(Khoản X, Điều Y - Tên văn bản)`.
5. **UI Rendering:** Hiển thị câu trả lời và cung cấp hộp thoại expander "Căn cứ pháp lý trích xuất" chứa toàn bộ trích dẫn của các văn bản nguồn.
