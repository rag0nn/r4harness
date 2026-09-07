"""
R4Agent API client kullanım örnekleri.

Çalıştırmak için önce server'ı başlatın:
    python -m r4agent.backend.server
"""
from r4agent.backend.client import R4Client
from r4agent.backend.structs import QueryStreamChunk

client = R4Client()


def query_stream(text: str):
    """Stream/non-stream sorguyu delta parçalarını doğrudan yazarak gösterir."""
    for chunk in client.query(text):
        if isinstance(chunk, QueryStreamChunk):
            print(chunk.content, end="", flush=True)
        else:
            print(chunk.content)
    print()


def add_document(title: str, content: str):
    response = client.rag_add_content(title, content)
    print(f"result={response.result}, error={response.error}")


def list_documents():
    response = client.rag_list_elements()
    if response.result:
        for elem in response.elements:
            print(elem)
    else:
        print(f"error={response.error}")


def remove_document(title: str):
    response = client.rag_remove_elements(title)
    print(f"result={response.result}, error={response.error}")


def query_loop():
    while True:
        text = input("\n--> ")
        if text == "exit":
            break
        print("<------->")
        query_stream(f"{text} text")


if __name__ == "__main__":
    # add_document("deneme", "Deneme içeriği")
    # list_documents()
    # remove_document("deneme")
    # query_stream("Şuan saat kaç?")
    query_loop()