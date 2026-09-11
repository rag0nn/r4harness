from r4agent import R4Agent, Message
from r4agent.struct import Roles
import pytest
from pathlib import Path

@pytest.fixture
def r4():
    r4 = R4Agent()
    return r4


class TestR4Agent:

    def test_send(self, r4:R4Agent):
        response = r4.send("Türkiyenin ilk cumhurbaşkanı kimdir?")
        content, thinking = "", ""
        for cnt, thk in response:
            content += cnt
            thinking += thk
            
    def test_messages_save_reset_load(self, r4: R4Agent, tmp_path):
        # tmp_path benzersiz bir Path objesi üretir
        r4.message_sequnce.add(Message(Roles.user,content="Test Query"))
        r4.message_sequnce.add(Message(Roles.assistant,content="Test Response"))
        saved_path = r4.save_messages(str(tmp_path))

        # Dosyanın oluştuğunu ve içeriğini doğrulayın
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        
        r4.reset_messages()
        assert len(r4.message_sequnce) <= 1
        
        r4.load_messages(str(saved_path))
        assert len(r4.message_sequnce) == 3

    
    
        