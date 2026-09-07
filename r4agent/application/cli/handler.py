
from pathlib import Path
from r4agent import R4Agent, Message, MessageSequence
from datetime import datetime

class Handler:
    
    CHAT_DIRECTORY = Path(__file__).parent / "chats"

    def __init__(self, r4: R4Agent):
        self.CHAT_DIRECTORY.mkdir(exist_ok=True)
        self.r4 = r4
    
    def list_chats(self)->list[Path]:
        paths =  [i for i in self.CHAT_DIRECTORY.iterdir()]  
        return paths
        
    def save_chat(self):
        self.r4.save_messages(self.CHAT_DIRECTORY)
    
    def load_chat(self, path):
        self.r4.load_messages(path)
    
    def reset_chat(self):
        self.r4.reset_messages()
    
    