import pytest
from r4agent.rag import RAGClient, Chunker, QDrantDatabase, VectorDatabaseConfig, ChunkerConfig
from r4agent.struct.models import CosmosEmbedding, CosmosConfig
@pytest.fixture
def client():
    embedding = CosmosEmbedding(CosmosConfig())
    chunker = Chunker(ChunkerConfig())
    db = QDrantDatabase(VectorDatabaseConfig())
    rag_client = RAGClient(db, chunker, embedding)
    
    yield rag_client
    
    # Testler bittiğinde otomatik kapanması için teardown eklemek daha güvenlidir
    rag_client.close()


class TestRAGClient:
    
    def test_add_document(self, client: RAGClient, tmp_path):
        md = "# Test-document\nLong test content in this test document"
        
        # Pytest'in güvenli geçici dosya yapısı
        temp_file = tmp_path / "test.md"
        temp_file.write_text(md, encoding="utf-8")
            
        client.add_document(str(temp_file))

    def test_add_content(self, client: RAGClient):
        for i in range(7):
            client.add_content(f"test-content-{i}", "test-title")

    def test_query_and_remove(self, client: RAGClient):
        # Bağımsız test ilkesi: Önce içerik ekle, sorgula ve silmeyi doğrula
        client.add_content("test-content-query", "test-title")
        
        elems = client.query("test-content-query")
        assert len(elems) > 0
        
        # ID ile silme
        client.remove_elements([elem.id for elem in elems[:1]])
        
        # Doküman başlığı ile silme
        client.remove_elements("test-title")

    def test_list_elements(self, client: RAGClient):
        elements = client.list_elements()
        assert elements is not None