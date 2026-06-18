import os
import json
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
from fastembed import TextEmbedding

# Đảm bảo in unicode ra console không bị lỗi trên Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

class LegalRetriever:
    def __init__(self, collection_name="legal_corpus", db_path="./qdrant_db"):
        self.collection_name = collection_name
        self.db_path = db_path
        
        print(f"Đang khởi tạo Qdrant Client (chế độ local)... Cơ sở dữ liệu lưu tại '{os.path.abspath(db_path)}'")
        # Chạy local mode, lưu ra đĩa cứng ở thư mục db_path
        self.client = QdrantClient(path=db_path)
        
        # Lấy danh sách mô hình được hỗ trợ trong môi trường hiện tại
        supported_models = [m["model"] for m in TextEmbedding.list_supported_models()]
        
        # Chọn mô hình phù hợp nhất: Ưu tiên bge-m3, nếu không có thì dùng paraphrase-multilingual-MiniLM-L12-v2 cho Tiếng Việt
        if "BAAI/bge-m3" in supported_models:
            self.model_name = "BAAI/bge-m3"
        elif "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" in supported_models:
            self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        else:
            self.model_name = None  # Sẽ dùng model mặc định của fastembed
            
        if self.model_name:
            print(f"Khởi tạo mô hình nhúng '{self.model_name}'...")
            self.embedding_model = TextEmbedding(model_name=self.model_name)
        else:
            print("Khởi tạo mô hình nhúng mặc định của FastEmbed...")
            self.embedding_model = TextEmbedding()
            self.model_name = self.embedding_model.model_name
            
        # Tự động phát hiện số chiều vector của mô hình nhúng
        dummy_vector = next(self.embedding_model.embed(["test"]))
        self.vector_size = len(dummy_vector)
        print(f"Đã sẵn sàng. Model: {self.model_name} | Vector dimension: {self.vector_size}")
        
        # Đảm bảo collection tồn tại
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Tạo collection nếu chưa tồn tại trong cơ sở dữ liệu."""
        try:
            if not self.client.collection_exists(self.collection_name):
                print(f"Đang tạo collection mới '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                print(f"Đã tạo collection '{self.collection_name}' thành công.")
        except Exception as e:
            print(f"Lỗi khi kiểm tra hoặc tạo collection: {e}")

    def ingest_data(self, json_path="legal_corpus.json", batch_size=64):
        """Đọc file JSON, tạo embeddings và nạp điểm (points) vào Qdrant."""
        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Không tìm thấy file JSON nguồn tại: {json_path}")
                
            print(f"Đang đọc dữ liệu từ: {json_path}...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            print(f"Tổng số phần tử cần nạp: {len(data)}")
            
            # Xóa collection cũ nếu đã tồn tại để đảm bảo nạp dữ liệu sạch, không trùng lặp
            print(f"Đang làm sạch collection '{self.collection_name}' trước khi nạp...")
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            
            # Nạp dữ liệu theo từng batch để tối ưu hiệu suất
            for idx in range(0, len(data), batch_size):
                batch = data[idx:idx + batch_size]
                texts = [item["text"] for item in batch]
                
                # Tạo embeddings cho toàn bộ text trong batch
                embeddings = list(self.embedding_model.embed(texts))
                
                points = []
                for j, item in enumerate(batch):
                    point_id = idx + j  # Sử dụng chỉ mục làm ID của Point
                    
                    # Chuyển đổi vector numpy sang list float để Qdrant nhận diện
                    vector_list = embeddings[j].tolist()
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=vector_list,
                        payload={
                            "text": item["text"],
                            "source_file": item["metadata"]["source_file"],
                            "doc_type": item["metadata"]["doc_type"],
                            "chuong": item["metadata"]["chuong"],
                            "dieu": item["metadata"]["dieu"],
                            "ten_dieu": item["metadata"]["ten_dieu"]
                        }
                    ))
                
                # Thực hiện upsert points lên Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"-> Đã nạp thành công các point từ {idx} đến {idx + len(batch) - 1}")
                
            print("✓ Quá trình nạp dữ liệu hoàn tất thành công!")
            
        except Exception as e:
            print(f"Lỗi trong quá trình nạp dữ liệu (ingest_data): {e}")

    def search(self, query, top_k=15, doc_type_filter=None):
        """Tìm kiếm ngữ nghĩa trong cơ sở dữ liệu Qdrant có hỗ trợ lọc doc_type."""
        try:
            # Sinh vector cho câu truy vấn
            query_vector = next(self.embedding_model.embed([query])).tolist()
            
            # Xây dựng bộ lọc điều kiện (Filter) nếu có yêu cầu lọc loại văn bản
            query_filter = None
            if doc_type_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="doc_type",
                            match=MatchValue(value=doc_type_filter)
                        )
                    ]
                )
                
            # Thực hiện tìm kiếm trên Qdrant bằng query_points
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            
            # Định dạng kết quả đầu ra
            formatted_results = []
            for hit in search_result.points:
                formatted_results.append({
                    "text": hit.payload.get("text"),
                    "metadata": {
                        "source_file": hit.payload.get("source_file"),
                        "doc_type": hit.payload.get("doc_type"),
                        "chuong": hit.payload.get("chuong"),
                        "dieu": hit.payload.get("dieu"),
                        "ten_dieu": hit.payload.get("ten_dieu")
                    },
                    "score": hit.score
                })
            return formatted_results
            
        except Exception as e:
            print(f"Lỗi trong quá trình tìm kiếm (search): {e}")
            return []

if __name__ == "__main__":
    # Khởi tạo retriever và nạp dữ liệu từ file JSON vào Qdrant DB khi chạy trực tiếp file này
    db = LegalRetriever()
    db.ingest_data("legal_corpus.json")

