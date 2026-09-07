# R4Agent Core Mimarisi

## Katmanlar

```text
CLI/TUI
  -> Handler / SessionController
  -> R4Agent
  -> Provider registry + MCP client
  -> Model adapters / RAG / Qdrant

HTTP API
  -> backend/server.py
  -> R4Agent ve RAG servisleri
  -> providers.py
  -> model, embedding ve database adapter'ları
```

- `struct/`: Mesaj, konuşma geçmişi ve provider arayüzleri gibi temel sözleşmeleri taşır.
- `master.py`: Agent orchestration katmanıdır. Kullanıcı mesajını geçmişe ekler, tool loop'u çalıştırır ve context model çıktısını stream eder.
- `providers.py`: Model, embedding, Whisper ve database nesnelerini lazy oluşturur. Registry seçimini ve mevcut provider kaynaklarının rebuild akışını yönetir.
- `tools/`: MCP sunucusuyla iletişimi sağlar. MCP işlemleri ayrı bir asyncio thread'i ve istek kuyruğu üzerinden yürütülür.
- `rag/`: Doküman parse etme, chunk'lama, embedding üretme ve Qdrant işlemlerini kapsar.
- `backend/`: HTTP endpoint sözleşmeleri ve backend client'ını içerir.
- `application/cli/`: Textual TUI ve CLI session adapter'larını içerir. Core orchestration burada yeniden uygulanmaz.

## Agent akışı

1. `R4Agent` provider registry üzerinden context, tool ve embedding modellerini alır.
2. MCP oturumundan tool talimatlarını ve tool şemalarını edinir.
3. Kullanıcı mesajını `MessageSequence` içine ekler.
4. Tool modeli en fazla sınırlı sayıda tur boyunca çağrı seçer.
5. MCP tool sonuçları konuşma geçmişine `tool` mesajı olarak eklenir.
6. Context modeli son geçmiş üzerinden cevap üretir.
7. Stream çıktısı tüketilir ve tamamlanan assistant mesajı geçmişe yazılır.

`MessageSequence` mutable konuşma state'idir. Backend query route'u her istek için ayrı `R4Agent` ve dolayısıyla ayrı konuşma geçmişi oluşturur; provider/model nesneleri paylaşılabilir olsa da konuşma ve tool oturumu izolasyonu korunur.

## Provider yaşam döngüsü

Provider seçimleri `Registery` üzerinden tutulur. `providers.py` nesneleri ilk erişimde lazy oluşturur ve kilit altında cache'ler. `rebuild()` mevcut database kaynağını kapatıp cache'leri temizler; sonraki erişim seçili provider'ları yeniden kurar.

Bu yapı CLI için pratiktir, ancak aktif backend istekleri sırasında rebuild yapılırsa yaşam döngüsü yarışı oluşabilir. Performans fazında rebuild için active-request drain, generation veya açık bir lifecycle manager uygulanmalıdır.

## RAG akışı

```text
kaynak belge/metin
  -> DocumentConverter
  -> Chunker
  -> embedding modeli
  -> Qdrant upsert

sorgu
  -> tek embedding
  -> Qdrant similarity query
  -> payload kayıtları
```

Listeleme ve similarity response'larında embedding vektörleri taşınmadığı için gereksiz payload ve bellek maliyeti azaltılır. Büyük ingest işlemleri için bir sonraki performans adımı batch embedding ve ölçümlü batch upsert'tir.

## Backend sınırı

`backend/server.py` endpoint wiring katmanıdır. Query, RAG ekleme/listeleme/silme ve provider rebuild işlemlerini dışarıya HTTP sözleşmesi olarak sunar. Senkron model, RAG ve lifecycle çağrıları async route'larda threadpool sınırında çalıştırılır; böylece uzun blocking işlemler event loop'u doğrudan durdurmaz.

Streaming protokolü delta chunk yaklaşımını kullanır: server yalnızca yeni content/thinking parçasını gönderir, client parçaları birleştirir. Bu, kümülatif response tekrarlarını ve gereksiz SSE payload'ını azaltır.

Senkron model, embedding, Qdrant ve provider rebuild çağrıları async route'larda `asyncio.to_thread` sınırında çalışır. Böylece blocking işlerin HTTP event loop'unu doğrudan durdurması engellenir.

## State sahipliği

- Konuşma geçmişi: agent/session scope
- Provider modelleri: process veya registry scope
- Qdrant client: provider/database scope
- MCP session: açık lifecycle scope
- HTTP request: geçici orchestration scope

Bu ayrım concurrency ve provider rebuild davranışının temelidir.

## Performans yaklaşımı

Öncelik sırası ölçümle ilerler:

1. Request izolasyonu ve event-loop blocking kontrolü
2. Delta streaming ile payload azaltma
3. Batch embedding ve Qdrant payload azaltma
4. Tool schema cache ve kontrollü lifecycle
5. Uzun konuşmalar için history budget/policy

Ölçülecek temel metrikler p50/p95 latency, time-to-first-token, output token/s, SSE byte miktarı, embedding süresi/çağrı sayısı, Qdrant süresi, CPU ve RSS'tir.

TUI bu core katmanlarını kullanır ancak agent/provider/RAG kurallarını yeniden tanımlamaz. TUI'nin kendi widget, controller ve command ayrımı `application/cli/summary.md` içinde açıklanır.
