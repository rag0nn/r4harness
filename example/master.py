from r4agent.utils import setup_logging
from r4agent import R4Agent

setup_logging(force=True)

STREAM = True

r = R4Agent(
    stream=STREAM
)

while True:
    inp = input(">")
    if inp == "exit":
        break
    if inp == "show":
        print(r.message_sequnce)
        break
    if not isinstance(inp, str):
        try:
            inp = str(inp)
        except Exception as e:
            raise ValueError(e)
        
    last_len = 0
    for cnt, thk in r.send(inp):
        # cnt'nin string olduğunu varsayarsak, sadece yeni gelen farkı yazdırın
        new_text = cnt[last_len:]
        print(new_text, end='', flush=True)
        last_len = len(cnt)
        if thk:
            print("Thinking: "+thk)
    print()  # Akış bitince alt satıra geçmek için
        