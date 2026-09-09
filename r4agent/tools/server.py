"""Küçük ama tam bir MCP sunucusu örneği.

Kurulum:
    python -m pip install "mcp[cli]>=2,<3"

MCP Inspector ile geliştirme:
    mcp dev r4harness/server.py

Doğrudan stdio transport ile çalıştırma:
    python r4harness/server.py

Streamable HTTP ile çalıştırma:
    python r4harness/server.py --transport streamable-http --port 8000

HTTP istemcileri bu durumda http://127.0.0.1:8000/mcp adresine bağlanır.
"""

from __future__ import annotations

import argparse
import re
from typing import Literal
from datetime import datetime
from pathlib import Path

from mcp.server import MCPServer
import httpx

mcp = MCPServer(
    name="r4agent",
    title="r4Agent MCP Server",
    version="1.0.0",
    instructions=( # LLM Prompt eklenen bilgilendirici metin
        "Harness MCP; "
        "Mevcut zaman, web'den site fetch "
        "vb. özel nitelikli işlemleri yapan tooları içerir."
    ),
    description=None, # insan kullanıcıları bilgilendirici metin.

)

import os as _os
from ..providers import ProviderManager as _ProviderManager, RegisterySet as _RegisterySet

_embed_model = _os.environ.get("R4AGENT_EMBED_MODEL", "cosmos")
_provider = _ProviderManager(_RegisterySet(embed_model=_embed_model))
@mcp.tool()
async def get_time() -> str:
    """Mevcut şuanki zamanı döndürür.

    Saat kaç? Zaman ne? Bugün günlerden ne? gibi soruları cevaplar
    """
    now = datetime.now()
    clock = now.strftime("%H:%M:%S")

    months = {
        1: "Ocak",
        2: "Şubat",
        3: "Mart",
        4: "Nisan",
        5: "Mayıs",
        6: "Haziran",
        7: "Temmuz",
        8: "Ağustos",
        9: "Eylül",
        10: "Ekim",
        11: "Kasım",
        12: "Aralık",
    }
    return f"{now.day:02d} {months[now.month]} {now.year} {clock}"

@mcp.tool()
async def fetch(url: str) -> str:
    """Bir http url'den veriyi çeker. Bir website kaynağından otomatik olarak veri öğrenmen gerektiğinde bunu kullanmalısın"""
    # Güvenlik ve performans için ti meout belirleyin
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            response = await client.get(url)
            # HTTP 4xx veya 5xx hatalarında exception fırlatır
            response.raise_for_status() 
            return response.text
        except httpx.HTTPStatusError as e:
            return f"HTTP Hatası: {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Bağlantı Hatası: {str(e)}"
        
@mcp.tool()
async def remote_blog_contents(url:str):
    """
        Uzak sunucuda içeriklerim var, bu tool blogumdan içerikleri görüntülemek için kullanılır. 
        Bu içerikler notlar, projeler ve gönderiler olarak görüntülenebilir.
        İçerikler gerekli url kombinasyonu ile elde edilebilir. Bu kombinasyonlar kullanılarak mevcut not içeriği,
        notların listesi, proje içeriği, projelerin listesi, gönderi içeriği, gönderilerin listesi görüntülenebilir.
        Anahter Kelimeler: Blog, BLOG, blog
        Kombinasyonlar:
            root (ana sayfa) : https://tarhannes.com.tr
            notes (notların listelendiği endpoint) : /notes
            projects (projelerin listelendiği endpoint): /projects
            posts (kapsamlı yazıların listelendiği endpoint): /posts
            {slug} (içerik, slug benzersiz içerik değişkeni, listelerden görüntülenebilir): /{slug}
        Sorgular:
            page (sayfaları görüntülemek, {X} sayfa sayısı): ?q=&page={X}    
    """
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            response = await client.get(url)
            # HTTP 4xx veya 5xx hatalarında exception fırlatır
            response.raise_for_status() 
            return response.text
        except httpx.HTTPStatusError as e:
            return f"HTTP Hatası: {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Bağlantı Hatası: {str(e)}"
            
@mcp.tool()
async def local_contents(query: str) -> str:
    """Yerel veritabanında arama yapıp ilgili içerikleri döndürür.

    Bu araç, depolanan belgeler içinde anahtar kelime veya benzer içerik araması
    gerektiğinde kullanılır. Arama sonucu, belge adı ve içeriği formatlanarak tek
    bir metin olarak geri döner.

    Args:
        query: Aranacak metin veya anahtar kelime.

    Returns:
        Bulunan belgelerden oluşturulmuş metin çıktısı.
    """
    try:
        elems = _provider.dbclient.query(content=query)
        text = ""
        for elem in elems:
            text += f"document: {elem.document} content: {elem.content}"
        return text
    except Exception as e:
        return f"local_contents hatası: {type(e).__name__}: {e}"

@mcp.tool()
async def read_code(paths: list[str]) -> str:
    """Belirtilen dosya yollarındaki içerikleri okur ve tek bir çıktıya birleştirir.

    Bu araç, kod veya metin dosyalarının içeriğini hızlıca incelemek için kullanılır.
    Birden fazla dosya verildiğinde tüm dosyaların içeriğini tek metin olarak döner.

    Args:
        paths: Okunacak dosya yolları listesi.

    Returns:
        "--- dosya_yolu ---\n...içerik..." formatında birleştirilmiş çıktı.
        Dosya bulunamazsa veya okunamazsa bununla ilgili uyarı eklenir.
    """
    contents: list[str] = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = file.read()
            contents.append(f"--- {path} ---\n{data}")
        except FileNotFoundError:
            contents.append(f"--- {path} ---\n[Dosya bulunamadı]")
        except OSError as exc:
            contents.append(f"--- {path} ---\n[Okuma hatası: {exc}]")

    return "\n\n".join(contents)

@mcp.tool()
async def find_file(pattern: str, root: str = ".") -> str:
    """Regex ile dosya adına veya göreli dosya yoluna göre dosya arar.

    Arama, varsayılan olarak mevcut çalışma dizininden başlayarak alt dizinlerde
    recursive yapılır. Regex, dosyanın çalışma dizinine göre göreli yolu üzerinde
    çalıştırılır; böylece hem dosya adı hem de klasör yolu eşleştirilebilir.

    Args:
        pattern: Dosya adı veya göreli dosya yoluyla eşleşecek regex deseni.
            Örneğin ``r"\\.py$"`` tüm Python dosyalarını bulur.
        root: Aramanın başlayacağı dizin. Varsayılan değer mevcut çalışma
            dizinidir.

    Returns:
        Eşleşen dosya yollarını satır satır döndürür. Hiç eşleşme yoksa bilgi
        mesajı döner; geçersiz regex veya okunamayan kök dizin için hata mesajı
        döner.
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Geçersiz regex: {exc}"

    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return f"Arama dizini bulunamadı: {root}"

    matches: list[str] = []
    for path in root_path.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(root_path).as_posix()
            if regex.search(relative_path):
                matches.append(relative_path)

    if not matches:
        return f"Regex ile eşleşen dosya bulunamadı: {pattern}"

    return "\n".join(sorted(matches))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r4agent MCP sunucusu")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport türü (varsayılan: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP transport için dinlenecek adres (varsayılan: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP transport portu (varsayılan: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transport: Literal["stdio", "streamable-http"] = args.transport

    if transport == "streamable-http":
        mcp.run(transport=transport, host=args.host, port=args.port)
        return

    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
