from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Generator, Any, List, Tuple, Union



# == General Use Structs ==========================

@dataclass(frozen=True)
class Roles:
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"

@dataclass
class Message:
    role:str
    content:str
    tool_calls: list | None = None

    def to_dict(self):
        result = {k: v for k, v in asdict(self).items() if v is not None}
        if result.get("tool_calls"):
            result["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in result["tool_calls"]
            ]
        return result
    
    def __repr__(self):
        return f"role: {self.role} content: {self.content} tool_calls: {self.tool_calls}"
        
class MessageSequence:
    
    def __init__(self, initial_system_prompt:str):
        self.initial_system_prompt = initial_system_prompt
        self.sequence:list[Message] = [
            Message(role=Roles.system, content=initial_system_prompt)
        ]
        
    def __len__(self):
        return len(self.sequence)
        
    def __str__(self):
        return "\n\n".join([f"role: {e.role} content: {f"{e.content[:20]}..." if e.content else "-"} tools_calls: {f"{e.tool_calls[:10]}..." if e.tool_calls else "-"}" for e in self.sequence])

    def add(self, message:Message):
        """Mesajı konuşma sırasına ekler ve debug kaydı üretir."""
        self.sequence.append(message)
        logging.info(f"[{self.__class__.__name__}] Added new message => {message}")
        
    def reset(self):
        """Konuşmayı başlangıç system mesajını koruyarak temizler."""
        self.sequence = [
            Message(role=Roles.system, content=self.initial_system_prompt)
        ]
        logging.info(f"[{self.__class__.__name__}] Mesaj kuyruğu başarıyla temizlendi")
        
    def get_as_dicts(self)->list[dict]:
        return [m.to_dict() for m in self.sequence]
    
    def save(self, path:str | Path) -> Path:
        """Konuşma geçmişini JSON dosyası olarak kalıcılaştırır. 
            Args:
                path (Path): kaydedilecek klasörün yolu
            Returns:
                Kaydedilmiş tam yol.
        """
        if isinstance(path, str):
            path = Path(path)
        
        unique = datetime.now().strftime("%d%m%Y_%H%M%S")
        file_path = path / f"{unique}.json"
        data = self.get_as_dicts()
        
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Mesaj kuyruğu {file_path} konumuna kaydedildi")
        return file_path
    
    def load(self, path: str | Path):
        """JSON geçmişini doğrulayıp mevcut konuşma sırasının yerine yükler."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Bu dosya bulunamadı: {path}")

        self.reset()
        data = json.loads(path.read_text(encoding="utf-8"))
        self.sequence = [
            Message(role=m["role"], content=m["content"], tool_calls=m.get("tool_calls"))
            for m in data
        ]
        if self.sequence and self.sequence[0].role == Roles.system:
            self.initial_system_prompt = self.sequence[0].content

        logging.info(f"Mesaj kuyruğu {path} konumundan yüklendi")
        return self
    
# == Abstaction Structs ==========================
    
class BaseContextGenerationModel(ABC):

    def __init__(self):
        super().__init__()
            
    @abstractmethod
    def send(
        self, 
        message_sequence: MessageSequence, 
        stream: bool = False, 
        tools: list | None = None
    ) -> Union[
        Tuple[str, str], 
        Generator[Tuple[str, str], None, None
    ]]:
        """Chats with generation model
        Inputs:
            sequnce (MessageSequence): Tüm mesaj sekansını alır ve doğru bir string ifade olarak modele verir.
                Mesajlardaki tool_calls alanları genel (name, params) tuple formatındadır;
                model adaptörü bunları kendi formatına çevirir.
        Returns:
            message (str): Cevap mesajı içeriği
            thinking (str): Eğer model thinking yaptıysa onun içeriği 
            tool calls (tuple[str,dict]): name ve paramsları tutan listeyi döndürür.
        """
        
class BaseToolGenerationModel(ABC):
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def send(
        self, 
        query: str, 
        tools: list | None = None
    ) ->List[Any]:
        """
        Chats with tool generation model.
        
        Input:
            query (str): İstenen görev sorgusu
            tools (list): Görevleri gerçekleştiren toolların listesi
        Returns:
            tool calls model (tuple[str,dict]): model format türünde name ve paramsları tutan listeyi döndürür.
            tool calls mcp (tuple[str,dict]): mcp türünde name ve paramsları tutan listeyi döndürür.
        """
        
class BaseEmbeddingGenerationModel(ABC):
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def embed(self, text:str)->list[float]:
        """Girilen yazının embedding vektörünü döndürür."""