from r4agent import FasterWhisper, FasterWishperConfig, R4Agent
from r4agent.utils import setup_logging
import time

setup_logging(force=True)

# == Import ==========================
whisper = FasterWhisper(FasterWishperConfig())
r4 = R4Agent(stream=True)

# == Record ==========================
# print("Start the record...")
whisper.start_recording()
time.sleep(5)
segments = whisper.stop_recording()
# print("Record closed")

# == Execution ==========================
words = []
for segment in segments:
    words.append(segment.text)
content = " ".join([w for w in words])
# print("İçerik: ",content)

last_len = 0

for cnt, thk in r4.send(content):
    # cnt'nin string olduğunu varsayarsak, sadece yeni gelen farkı yazdırın
    new_text = cnt[last_len:]
    print(new_text, end='', flush=True)
    last_len = len(cnt)
    if thk:
        print("Thinking: "+thk)
        
print()