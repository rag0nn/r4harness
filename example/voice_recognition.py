# from r4agent.struct.models import FasterWhisper, FasterWishperConfig
from r4agent.utils import setup_logging
from r4agent.providers import WHISPER
import time

setup_logging(force=False)

whisper = WHISPER

# == Send ile ==========================
# result = whisper.send(record_from_microphone=True, duration=5)

# # == Dinamik ==========================
whisper.start_recording()
time.sleep(8)
segments = whisper.stop_recording()
for segment in segments:
    print(segment.text)