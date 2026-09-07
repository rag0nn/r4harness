# TUI Mimarisi

## Ana akış

- `tui.py`: Textual uygulamasını kurar, ekran geçişlerini ve worker olaylarını koordine eder.
- `widgets.py`: Prompt, mesaj, tool zinciri, model banner ve log handler gibi görsel bileşenleri içerir.
- `command_parser.py`: Slash komutlarının ve prompt içindeki son komut parçasının çözümlemesini yapar.
- `message_renderer.py`: `MessageSequence` verisini chat ekranındaki Textual widget'larına dönüştürür.
- `query_controller.py`: Sorgu generation token'larını, stream tüketimini ve cooperative cancellation akışını yönetir.
- `log_controller.py`: Python logging kayıtlarını sınırlı bir buffer ile Logs ekranına aktarır ve uygulama kapanınca eski handler'ları geri yükler.
- `session_controller.py`: Chat listeleme, kaydetme, yükleme ve sıfırlama işlemlerini TUI'den ayırır.
- `handler.py`: TUI ile `R4Agent` arasındaki mevcut session adapter'ıdır.
- `styles.tcss`: Widget kimlikleri ve sınıfları için görsel yerleşim kurallarını taşır.

## Yaklaşım

TUI, agent/provider/RAG davranışını uygulamaya çalışmak yerine bu katmanları worker callback'leri üzerinden kullanır. Yeni modüller Textual ekranını doğrudan yönetmek yerine tek bir sorumluluğa odaklanır; böylece command parsing ve mesaj renderı gibi saf davranışlar bağımsız test edilebilir.

Worker event yaşam döngüsü ve ekranlar arası koordinasyon bilinçli olarak `tui.py` içinde bırakılmıştır. Bu dosya, Textual'ın event ve worker API'lerine bağlı composition root görevi görür; sorgunun saf stream/cancellation state'i ise `query_controller.py` içindedir. Core agent, provider, backend ve RAG dosyaları bu refactor kapsamında değiştirilmemiştir.
