from r4agent.utils import setup_logging
from r4agent.struct import OllamaConfig, OllamaGenModel, MessageSequence, Message, Roles
from pathlib import Path

if __name__ == "__main__":
    setup_logging()
    config = OllamaConfig()
    assistant = OllamaGenModel(config)
    message_sequence = MessageSequence("Sen bir yapay zeka asistanısın.")
    
    # == İlk Soru ==========================
    message_sequence.add(
        Message(role=Roles.user, content="Türkiyenin nüfusu kaçtır?")
        )
    content, thinking = assistant.send(message_sequence)
    message_sequence.add(Message(role="assistant", content=content))
    print(f"\n Response: {content}\n")
    
    # == İkinci Soru ==========================
    message_sequence.add(
        Message(
            role=Roles.user,
            content="Bu ülkenin cumhurbaşkanı kimdir?"
        )
    )
    content, thinking = assistant.send(message_sequence)
    message_sequence.add(Message(role="assistant", content=content))
    print(f"\n Response: {content}\n")
    
    # == Özet ==========================
    print("\n\nSummary")
    print(message_sequence.get_as_dicts())
    
    # == KAydet ==========================
    print("\n Save")
    message_sequence.save(path=Path(__file__).parent )
    
    # == Yükle ==========================
    print("\n Load")
    message_sequence.load(Path(__file__).parent / "22082026_191324.json" )
    
    print("\n\nSummary")
    for message in message_sequence.sequence:
        print(f"{message} \n ")
    
    # == Üçüncü Soru ==========================
    message_sequence.add(
        Message(
            role=Roles.user,
            content="Bu ülkede kaç bölge vardır?"
        )
    )
    content, thinking = assistant.send(message_sequence)
    message_sequence.add(Message(role="assistant", content=content))
    print(f"\n Response: {content}\n")
    
    # == Kaydet ==========================
    message_sequence.save(Path(__file__).parent)
    
