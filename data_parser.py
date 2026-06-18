import os
import re
import json
import sys
from docx import Document
from tqdm import tqdm

# Đảm bảo in unicode ra console không bị lỗi trên Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Các biểu thức chính quy để nhận diện cấu trúc văn bản pháp luật Việt Nam
RE_CHAPTER = re.compile(r'(?i)^Chương\s+([IVXLCDM]+)(?:\.|\:|\-)?\s*(.*)$')
RE_SECTION = re.compile(r'(?i)^Mục\s+(\d+)(?:\.|\:|\-)?\s*(.*)$')
RE_ARTICLE = re.compile(r'(?i)^Điều\s+(\d+)(?:\.|\:|\-)?\s*(.*)$')
RE_CLAUSE = re.compile(r'^\s*(\d+)\.\s+(.*)$')

def is_chapter(text):
    return bool(RE_CHAPTER.match(text))

def is_section(text):
    return bool(RE_SECTION.match(text))

def is_article(text):
    return bool(RE_ARTICLE.match(text))

def is_clause(text):
    return bool(RE_CLAUSE.match(text))

def get_doc_type(filename):
    """Phân loại loại văn bản dựa trên tên file."""
    filename_lower = filename.lower()
    if "luật" in filename_lower:
        return "Luật"
    elif "nghị định" in filename_lower:
        return "Nghị định"
    return "Khác"

def parse_docx(file_path):
    """Phân tích cú pháp file docx thành các chunk có cấu trúc."""
    filename = os.path.basename(file_path)
    doc_type = get_doc_type(filename)
    
    doc = Document(file_path)
    # Lấy toàn bộ paragraph không rỗng và thay thế non-breaking space
    paragraphs = []
    for p in doc.paragraphs:
        txt = p.text.replace('\xa0', ' ').strip()
        if txt:
            paragraphs.append(txt)
            
    chunks = []
    current_chuong_base = None
    current_muc = None
    current_dieu = None
    current_ten_dieu = None
    
    # State machine để tích lũy chunk
    current_chunk = None
    
    def save_current_chunk():
        nonlocal current_chunk
        if current_chunk and current_chunk['text']:
            # Gộp các dòng nội dung lại bằng ký tự xuống dòng
            full_text = "\n".join(current_chunk['text'])
            
            # Prepend tiêu đề Điều vào mỗi Khoản để làm phong phú ngữ cảnh RAG (Context Enrichment)
            if current_chunk.get("type") == "clause" and current_dieu:
                dieu_prefix = f"{current_dieu}"
                if current_ten_dieu:
                    dieu_prefix += f". {current_ten_dieu}"
                full_text = f"{dieu_prefix}\n{full_text}"
                
            # Chỉ lưu các chunk có text thực sự
            if full_text.strip():
                chunks.append({
                    "text": full_text,
                    "metadata": current_chunk['metadata']
                })
            current_chunk = None

    i = 0
    num_paragraphs = len(paragraphs)
    
    # Sử dụng tqdm để theo dõi tiến trình của từng paragraph trong file
    pbar = tqdm(total=num_paragraphs, desc=f"Đang xử lý {filename[:25]:<25}", leave=False)
    
    while i < num_paragraphs:
        p = paragraphs[i]
        
        # 1. Phát hiện Chương (Chapter)
        if is_chapter(p):
            save_current_chunk()
            
            match = RE_CHAPTER.match(p)
            roman = match.group(1)
            title = match.group(2).strip()
            
            # Lookahead: Nếu tiêu đề Chương nằm ở dòng kế tiếp
            if not title and i + 1 < num_paragraphs:
                next_p = paragraphs[i+1]
                if not is_chapter(next_p) and not is_section(next_p) and not is_article(next_p) and not is_clause(next_p):
                    title = next_p
                    i += 1
                    pbar.update(1)
            
            current_chuong_base = f"Chương {roman}"
            if title:
                current_chuong_base += f": {title}"
            
            # Reset Mục khi sang Chương mới
            current_muc = None
            i += 1
            pbar.update(1)
            continue
            
        # 2. Phát hiện Mục (Section)
        elif is_section(p):
            save_current_chunk()
            
            match = RE_SECTION.match(p)
            num = match.group(1)
            title = match.group(2).strip()
            
            # Lookahead: Nếu tiêu đề Mục nằm ở dòng kế tiếp
            if not title and i + 1 < num_paragraphs:
                next_p = paragraphs[i+1]
                if not is_chapter(next_p) and not is_section(next_p) and not is_article(next_p) and not is_clause(next_p):
                    title = next_p
                    i += 1
                    pbar.update(1)
            
            current_muc = f"Mục {num}"
            if title:
                current_muc += f": {title}"
                
            # Đánh dấu thoát khỏi điều cũ
            current_dieu = None
            current_ten_dieu = None
            
            i += 1
            pbar.update(1)
            continue
            
        # 3. Phát hiện Điều (Article)
        elif is_article(p):
            save_current_chunk()
            
            match = RE_ARTICLE.match(p)
            dieu_num = f"Điều {match.group(1)}"
            dieu_title = match.group(2).strip()
            
            # Lookahead: Nếu tên Điều nằm ở dòng kế tiếp
            if not dieu_title and i + 1 < num_paragraphs:
                next_p = paragraphs[i+1]
                if not is_chapter(next_p) and not is_section(next_p) and not is_article(next_p) and not is_clause(next_p):
                    dieu_title = next_p
                    i += 1
                    pbar.update(1)
            
            current_dieu = dieu_num
            current_ten_dieu = dieu_title if dieu_title else None
            
            # Xây dựng chuỗi thông tin Chương (ghép cả Mục nếu có)
            chuong_meta = current_chuong_base
            if current_chuong_base and current_muc:
                chuong_meta = f"{current_chuong_base} - {current_muc}"
                
            # Tạo chunk đại diện cho Điều (tiêu đề + nội dung ban đầu)
            chunk_header = p
            if dieu_title and p == dieu_num:
                chunk_header = f"{dieu_num}. {dieu_title}"
                
            current_chunk = {
                "text": [chunk_header],
                "type": "article",
                "metadata": {
                    "source_file": filename,
                    "doc_type": doc_type,
                    "chuong": chuong_meta,
                    "dieu": current_dieu,
                    "ten_dieu": current_ten_dieu
                }
            }
            i += 1
            pbar.update(1)
            continue
            
        # 4. Phát hiện Khoản (Clause)
        elif is_clause(p) and current_dieu is not None:
            save_current_chunk()
            
            chuong_meta = current_chuong_base
            if current_chuong_base and current_muc:
                chuong_meta = f"{current_chuong_base} - {current_muc}"
                
            current_chunk = {
                "text": [p],
                "type": "clause",
                "metadata": {
                    "source_file": filename,
                    "doc_type": doc_type,
                    "chuong": chuong_meta,
                    "dieu": current_dieu,
                    "ten_dieu": current_ten_dieu
                }
            }
            i += 1
            pbar.update(1)
            continue
            
        # 5. Dòng nội dung bình thường
        else:
            # Chỉ ghi nhận nếu đã ở trong một Điều luật cụ thể
            if current_dieu is not None:
                if current_chunk is None:
                    # Nếu chưa có chunk chạy ngầm, tạo mới dạng article
                    chuong_meta = current_chuong_base
                    if current_chuong_base and current_muc:
                        chuong_meta = f"{current_chuong_base} - {current_muc}"
                        
                    current_chunk = {
                        "text": [p],
                        "type": "article",
                        "metadata": {
                            "source_file": filename,
                            "doc_type": doc_type,
                            "chuong": chuong_meta,
                            "dieu": current_dieu,
                            "ten_dieu": current_ten_dieu
                        }
                    }
                else:
                    current_chunk["text"].append(p)
            i += 1
            pbar.update(1)
            
    save_current_chunk()
    pbar.close()
    return chunks

def main():
    data_dir = "data"
    output_file = "legal_corpus.json"
    
    # Tìm kiếm các file docx trong thư mục data
    if not os.path.exists(data_dir):
        print(f"Lỗi: Thư mục '{data_dir}' không tồn tại.")
        return
        
    docx_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".docx")]
    
    if not docx_files:
        print(f"Lỗi: Không tìm thấy file .docx nào trong thư mục '{data_dir}'.")
        return
        
    print(f"Tìm thấy {len(docx_files)} file docx cần xử lý.")
    all_chunks = []
    
    # Vòng lặp chính có in thanh tiến trình của từng file
    for file_path in docx_files:
        chunks = parse_docx(file_path)
        all_chunks.extend(chunks)
        print(f"✓ Hoàn thành {os.path.basename(file_path)}: Đã trích xuất {len(chunks)} chunks.")
        
    # Ghi toàn bộ dữ liệu ra file JSON duy nhất
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    print("\n=========================================")
    print(f"Hoàn thành phân tích toàn bộ tài liệu!")
    print(f"Tổng số chunk đã trích xuất: {len(all_chunks)}")
    print(f"Kết quả được lưu tại: {os.path.abspath(output_file)}")
    print("=========================================")

if __name__ == "__main__":
    main()
