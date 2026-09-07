from typing import Generator, Tuple, List, Any
from ollama import Client, ChatResponse
import logging

from .base import *
from ..tools.utils import ToolConvertionsOllama
from .model_configs import *
import os
import numpy as np
from ..utils import log_execution_time

# == Ollama ==========================
class OllamaGenModel(BaseContextGenerationModel):

    @log_execution_time
    def __init__(self, config: OllamaConfig):
        super().__init__()
        self.config = config
        self.client = Client()

    def send(
        self, 
        message_sequence: MessageSequence, 
        stream: bool = False, 
    ) -> Generator[Tuple[str, str], None, None]:
        response = self.client.chat(
            model=self.config.model,
            messages=message_sequence.get_as_dicts(),
            think=self.config.think,
            stream=stream,
        )

        if not stream:
            if isinstance(response, ChatResponse) and not response.done:
                logging.error("Cevap oluşturulamadı.")
                return "Cevap bir hata sebebiyle oluşturulamadı", "", []
                
            response_message = response.message.content or ""
            response_thinking = response.message.thinking or ""
            
            logging.info(f"Cevap başarıyla oluşturuldu. {response_message[:10]}...")
            yield response_message, response_thinking

        else:
            for chunk in response:
                response_message = chunk.message.content or ""
                response_thinking = chunk.message.thinking or ""
                yield response_message, response_thinking

class OllamaToolGenModel(BaseToolGenerationModel):
    
    @log_execution_time
    def __init__(self, config: OllamaConfig):
        super().__init__()
        self.config = config
        self.client = Client()
        
    def send(
        self, 
        messages: MessageSequence, 
        tools: list | None = None
    ) ->Tuple[List[Any],List[Any]]:
        tools = ToolConvertionsOllama.from_mcp(tools) if tools else None

        response = self.client.chat(
            model=self.config.model,
            messages=messages.get_as_dicts(),
            think=self.config.think,
            stream=False,
            tools=tools
        )
        if isinstance(response, ChatResponse) and not response.done:
            logging.error("Cevap oluşturulamadı.")
            return "Cevap bir hata sebebiyle oluşturulamadı", "", []
        
        tool_calls = response.message.tool_calls or []
        
        logging.info(f"Çağrılacak tool sayısı => {len(tool_calls)}...")
        return tool_calls, ToolConvertionsOllama.to_mcp(tool_calls)
    
class OllamaEmbeddingGenModel(BaseEmbeddingGenerationModel):
    
    @log_execution_time
    def __init__(self, config:OllamaConfig):
        super().__init__()
        self.config = config
        self.client = Client()
        
    def embed(self, text: str) -> list[float]:
        response = self.client.embed(
            self.config.model,
            input=text
        )
        return list(response.embeddings[0])
    
# == Gemini ==========================
class GeminiGenModel(BaseContextGenerationModel):
    
    @log_execution_time
    def __init__(self, config: GeminiGenConfig):
        from google import genai
        super().__init__()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # ENV DOSYASI BAŞKA BİRYERDE INIT EDİLİR
        self.config = config

    def send(self, message_sequence: MessageSequence, stream=False) -> Generator[Tuple[str, str], None, None]:
        # Gemini 2.x / Official GenAI SDK standart formatı
        contents = []
        for msg in message_sequence.get_as_dicts():
            role = msg.get("role")
            if role == Roles.system:
                continue
                
            gemini_role = "model" if role in (Roles.assistant, "model") else "user"
            content_text = str(msg.get("content") or msg.get("parts") or "")
            
            if not content_text:
                continue
                
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content_text}]
            })

        # client.interactions yerine client.models kullanıyoruz
        if stream:
            response = self.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config={
                    "system_instruction": message_sequence.initial_system_prompt,
                    "max_output_tokens": self.config.max_output_tokens,
                    "stop_sequences": self.config.stop_sequences,
                    "seed": self.config.seed,
                }
            )
            for chunk in response:
                yield chunk.text or "", ""
        else:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config={
                    "system_instruction": message_sequence.initial_system_prompt,
                    "max_output_tokens": self.config.max_output_tokens,
                    "stop_sequences": self.config.stop_sequences,
                    "seed": self.config.seed,
                }
            )
            yield response.text or "", ""
            
class GeminiEmbedding(BaseEmbeddingGenerationModel):
    
    @log_execution_time
    def __init__(self, config: GeminiEmbedConfig):
        super().__init__()
        from google.genai.types import EmbedContentConfig
        from google import genai
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) # ENV DOSYASI BAŞKA BİRYERDE INIT EDİLİR
        self.config = config
        self.gemini_config = EmbedContentConfig(
            output_dimensionality=config.vector_size)

    def embed(self, content:str) -> list[float]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Content must be a non-empty string.")

        result = self.client.models.embed_content(
            model=self.config.model,
            contents=content,
            config=self.gemini_config,
        )
        if not result.embeddings or result.embeddings[0].values is None:
            raise RuntimeError("Gemini returned no embedding values.")

        return [float(value) for value in result.embeddings[0].values]
    
# == Cosmos ==========================
class CosmosEmbedding(BaseEmbeddingGenerationModel):
    
    @log_execution_time
    def __init__(self, config:CosmosConfig):
        super().__init__()
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "CosmosEmbedding kullanabilmek için gerekli kütüphaneler eksik. "
                "Lütfen 'pip install onnxruntime transformers' komutu ile yükleyin."
            )
                
        self.config = config
        # Tokenizer'ı ana klasörden yükleme
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        # ONNX Runtime Oturumunu bir kere başlatıp RAM'de tutma
        self.session = ort.InferenceSession(
            self.config.onnx_path, 
            providers=['CPUExecutionProvider']
        )

    def _mean_pooling(self, model_output, attention_mask):
        """Hidden state çıktılarını tek bir vektöre (mean pooling) dönüştürür."""
        token_embeddings = model_output[0]  # Shape: (batch_size, seq_len, hidden_dim)
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

    def embed(self, text: str) -> list[float]:
        # 1. Metni tokenize etme
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=self.config.vector_size, 
            return_tensors="np"
        )
        
        # ONNX giriş sözlüğünü hazırlama
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # 2. ONNX çıkarımı (Inference)
        outputs = self.session.run(None, onnx_inputs)
        
        # 3. Mean Pooling ile vektör çıkarma
        embedding = self._mean_pooling(outputs, onnx_inputs["attention_mask"])[0]
        
        # 4. L2 Normalizasyon (Vektör benzerlikleri için şarttır)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # float32 numpy array'ini standart Python float listesine dönüştürme
        return embedding.tolist()
    
# == Voice2Text Modeller ==========================

class FasterWhisper:
    
    @log_execution_time
    def __init__(self, config: FasterWishperConfig):
        from faster_whisper import WhisperModel
        self.config = config  
        self.model = WhisperModel(
            self.config.model_size, 
            device="cuda", 
            compute_type= self.config.compute_type
        )
        # Recording state
        self.is_recording = False
        self.audio_buffer = None
        self.stream = None
        self.sample_rate = 16000
    
    def send(self, audio_input: str = None, record_from_microphone: bool = False, duration: int = None):
        """
        Transcribe audio from file or microphone using Faster Whisper.
        
        Args:
            audio_input (str): Path to audio file. Required if record_from_microphone is False.
            record_from_microphone (bool): If True, record audio from microphone instead of reading from file.
            duration (int): Duration in seconds to record from microphone (only used if record_from_microphone=True).
                          If None, recording continues until stopped manually.
        
        Returns:
            segments: Transcribed segments with timestamps and text.
            audio_path: Path to audio file (only returned when record_from_microphone=True)
        """
        
        # Eğer mikrofondan kayıt yapılacaksa
        if record_from_microphone:
            try:
                import sounddevice as sd
                import soundfile as sf
                import tempfile
                
                # Kayıt parametreleri
                sample_rate = 16000
                channels = 1
                
                logging.info(f"Mikrofondan kaydı başlatılıyor{' ({} saniye)'.format(duration) if duration else '...'}")
                
                # Mikrofondan ses kaydı
                audio_data = sd.rec(
                    int(sample_rate * duration) if duration else sample_rate * 60,  # 60 saniye default
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='float32'
                )
                sd.wait()  # Kaydın bitmesini bekle
                
                # Geçici bir dosyaya kaydet 
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_file.name, audio_data, sample_rate)
                audio_path = temp_file.name
                
                logging.info(f"Mikrofon kaydı tamamlandı: {audio_path}")
                
            except ImportError:
                raise ImportError(
                    "Mikrofondan kayıt yapabilmek için gerekli kütüphaneler eksik. "
                    "Lütfen 'pip install sounddevice soundfile' komutu ile yükleyin."
                )
        else:
            # Dosyadan okuma
            if audio_input is None:
                raise ValueError("audio_input parametresi sağlanmalı veya record_from_microphone=True olmalı.")
            
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_input}")
            
            audio_path = audio_input
            logging.info(f"Ses dosyası okunuyor: {audio_path}")
        
        # Transkripsiyon
        segments, info = self.model.transcribe(audio_path, language="tr")
        segments = list(segments)  # Convert generator to list to avoid exhausting it
        
        for segment in segments:
            logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        # Eğer geçici dosya kullanıldıysa, sil
        if record_from_microphone:
            try:
                os.remove(audio_path)
                logging.info(f"Geçici dosya silindi: {audio_path}")
            except Exception as e:
                logging.warning(f"Geçici dosya silinirken hata: {e}")
            
        return segments
    
    def start_recording(self):
        """UI'den başlat butonu ile kayıt başlatır"""
        import sounddevice as sd
        
        self.is_recording = True
        self.audio_buffer = []

        def audio_callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_buffer.append(indata.copy())
            if status:
                logging.warning(f"Ses kaydı hatası: {status}")
        
        self.stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=4096
        )
        self.stream.start()
        logging.info("Kayıt aktif")
    
    def stop_recording(self):
        """UI'den durdur butonu ile kayıt durdurur ve transkripsiyon yapar"""
        if not self.is_recording:
            logging.warning("Kayıt halihazırda çalışmıyor")
            return None
        
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        logging.info("Kayıt durduruldu")
        
        if not self.audio_buffer:
            logging.warning("Kayıtlı ses verisi yok")
            return None
        
        # Ses verilerini birleştir
        import numpy as np
        audio_data = np.concatenate(self.audio_buffer, axis=0)

        return self._transcribe_audio(audio_data)
    
    def _transcribe_audio(self, audio_data):
        """Ses verisini transkripsiyon yapar"""
        import soundfile as sf
        import tempfile
        
        # Ses dosyasına yaz
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, self.sample_rate)
        
        logging.debug(f"Geçici ses dosyası oluşturuldu: {temp_file.name}")
        
        # Transkripsiyon
        segments, info = self.model.transcribe(temp_file.name, language="tr")
        segments = list(segments)
        logging.debug(f"Tespit edilen dil: {info.language} (Olasılık: {info.language_probability:.2f})")
        logging.debug(f"Toplam segment sayısı: {len(segments)}")
        
        for segment in segments:
            logging.info(f"[{segment.start:.2f} s -> {segment.end:.2f}s] {segment.text}")
        
        # Dosyayı sil
        try:
            os.remove(temp_file.name)
            logging.debug(f"Geçici dosya silindi: {temp_file.name}")
        except Exception as e:
            logging.warning(f"Geçici dosya silinirken hata: {e}")
        
        return segments
    
    
# == Serial ==========================
class Registery:
    context_models: dict[str, Tuple[BaseContextGenerationModel, BaseModel]] = {
        "ollama" : (OllamaGenModel, OllamaConfig),
        "gemini" : (GeminiGenModel, GeminiGenConfig)
    }
    embed_models: dict[str, BaseEmbeddingGenerationModel] = {
        "ollama" : (OllamaEmbeddingGenModel, OllamaConfig),
        "gemini" : (GeminiEmbedding, GeminiEmbedConfig),
        "cosmos" : (CosmosEmbedding, CosmosConfig)
    }
    toolgen_models: dict[str, BaseToolGenerationModel] = {
        "ollama" : (OllamaToolGenModel, OllamaConfig)
    }

    def __init__(self) -> None:
        self.context_key = "ollama"
        self.embed_key = "cosmos"
        self.toolgen_key = "ollama"

    @staticmethod
    def _set_key(current: str, key: str, available: dict) -> str:
        if key not in available:
            raise ValueError(
                f"{key} registery'de bulunan geçerli bir key değil. "
                f"Mevcut keyler: {available.keys()}"
            )
        return key

    def set_context_model(self, key: str) -> None:
        self.context_key = self._set_key(self.context_key, key, self.context_models)

    def set_embed_model(self, key: str) -> None:
        self.embed_key = self._set_key(self.embed_key, key, self.embed_models)

    def set_toolgen_model(self, key: str) -> None:
        self.toolgen_key = self._set_key(self.toolgen_key, key, self.toolgen_models)

    def get_context_model(self, key: str):
        try: 
            model, config = self.context_models[key]
            return model(config())
        except Exception as e:
            raise ValueError(f"{e}\n\n{key} registery'de bulunan geçerli bir key değil. Mevcut keyler: {self.context_models.keys()}") 
    
    def get_embed_model(self, key: str):
        try: 
            model, config = self.embed_models[key]
            return model(config())
        except Exception as e:
            raise ValueError(f"{e}\n\n{key} registery'de bulunan geçerli bir key değil. Mevcut keyler: {self.embed_models.keys()}") 
    
    
    def get_toolgen_model(self, key: str):
        try: 
            model, config = self.toolgen_models[key]
            return model(config())
        except Exception as e:
            raise ValueError(f"{e}\n\n{key} registery'de bulunan geçerli bir key değil. Mevcut keyler: {self.toolgen_models.keys()}") 
    