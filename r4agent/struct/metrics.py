from dataclasses import dataclass, asdict
from typing import Optional, Any

@dataclass
class PerformanceMetrics:
    """R4Agent Metrik Veri Yapısı Dokümantasyonu
    
    Bu sınıf, model üretimi, bağlam kullanımı ve sistem bileşenlerinin
    çalışma sürelerine dair canlı ve tamamlanmış metrikleri tutar.

    Alanlar (Fields):
    ----------------
    ttft_ms (float):
        Time-to-First-Token (İlk Token Ulaşım Süresi - milisaniye).
        Kullanıcı isteğinin başlatıldığı an ile ilk yanıt parçacığının (delta chunk)
        üretildiği an arasında geçen süredir. İçeriğinde ağ gecikmesi, RAG arama
        süresi (embedding + Qdrant), MCP araçlarının çalıştırma süresi ve modelin 
        girdi bağlamını işleme (Prompt Evaluation / Prefill) sürelerini barındırır.

    output_tps (float):
        Tokens Per Second (Saniyedeki Üretilen Token Hızı - tok/s).
        İlk token ekrana geldikten yanıt tamamlanana kadar geçen sürede modelin
        ürettiği akıcı çıktı hızını ifade eder. Hesaplamaya TTFT gecikmesi dahil 
        edilmez; böylece modelin saf generation performansı ölçülür.

    prompt_tokens (int):
        Girdi Token Sayısı (Input / Prompt Tokens).
        Son istekte modele gönderilen sistem yönergesi (System Prompt), RAG doküman 
        parçaları, MCP araç (tool) şemaları ve mevcut kullanıcı mesajının toplam
        token büyüklüğüdür.

    completion_tokens (int):
        Çıktı Token Sayısı (Output / Completion Tokens).
        Modelin aktif yanıt döngüsünde (veya araç çağrısında) ürettiği toplam 
        token sayısıdır.

    total_context_tokens (int):
        Toplam Aktif Bağlam Büyüklüğü (Total Context Size).
        Mevcut konuşma geçmişi (MessageSequence) ve son eklenen prompt dahil olmak 
        üzere modelin belleğinde tutulan aktif token miktarıdır.

    max_context_window (int):
        Maksimum Bağlam Penceresi Limiti (Max Context Limit).
        Aktif kullanılan modelin desteklediği en fazla token sınırıdır 
        (Örn: Ollama llama3 için 8192, Gemini için 1048576).

    tool_duration_ms (float):
        Araç Çağrı Gecikmesi (MCP / External Tool Execution Latency - ms).
        Model bir MCP aracı (ör. find_file, qdrant_search) çalıştırmaya karar verdiğinde,
        bu aracın dış süreçlerde çalışıp yanıtını döndürmesine kadar geçen süredir.

    total_duration_ms (float):
        Toplam Geçen Süre (Total Elapsed Time - ms).
        Bir mesajın oluşma süresidir; kullanıcı promptunun gönderildiği andan model
        yanıtının tamamlandığı ana kadar geçen toplam süreyi ifade eder.
    """
    ttft_ms: float = 0.0
    output_tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_context_tokens: int = 0
    max_context_window: int = 8192
    tool_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    @property
    def context_usage_pct(self) -> float:
        """Bağlam penceresinin doluluk oranını yüzde olarak verir."""
        if self.max_context_window <= 0:
            return 0.0
        return round((self.total_context_tokens / self.max_context_window) * 100, 2)

    @property
    def total_tokens(self) -> int:
        """İstekte toplam tüketilen token sayısı (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "PerformanceMetrics") -> "PerformanceMetrics":
        """Katmanlar arası birikimli toplam üretir.

        Token sayıları ve süreler toplanır; `output_tps` ortalaması alınır
        (üretim hızı bir oran olduğu için toplamı anlamsızdır) ve
        `max_context_window` için iki tarafın en büyüğü korunur.
        """
        if not isinstance(other, PerformanceMetrics):
            return NotImplemented
        return PerformanceMetrics(
            ttft_ms=self.ttft_ms + other.ttft_ms,
            output_tps=round(
                (self.output_tps + other.output_tps) / 2, 2
            ) if (self.output_tps or other.output_tps) else 0.0,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_context_tokens=self.total_context_tokens + other.total_context_tokens,
            max_context_window=max(self.max_context_window, other.max_context_window),
            tool_duration_ms=self.tool_duration_ms + other.tool_duration_ms,
            total_duration_ms=self.total_duration_ms + other.total_duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Wire/SSE formatına serileştirmek için sözlük döndürür."""
        return asdict(self)