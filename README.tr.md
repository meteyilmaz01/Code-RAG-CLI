# code-rag-cli

[English](README.md)

Terminalde çalışan, küçük ve **yerel öncelikli bir Code-RAG aracı**. Bir kod tabanını gösterirsin, düz dille
soru sorarsın, dayandığı dosya ve satır numaralarını belirterek cevap verir.

Amacı, bir kod soru-cevap hattının nasıl bir araya geldiğini (tarama, parçalama, embedding, hibrit arama,
cevap) gösteren kısa ve okunabilir bir referans olmak. Aider ya da Continue gibi olgun araçların alternatifi
olma iddiası yok.

```
$ coderag ask "Hangi klasörler indekslemede her zaman atlanıyor ve atlanan en büyük dosya boyutu nedir?"
✓ İndeks güncellendi (1 yeni, 1 güncellenen, 0 silinen dosya) (25.1 sn)
✓ Soru İngilizce arama sorgusuna çevriliyor (30.6 sn)
✓ Kod aranıyor (anlam + kelime): test/test_scanner.py: _paths, _write, ... (0.9 sn)
✓ Sonuçlar RRF ile birleştiriliyor
✓ LLM cevabı yazıyor (5 kod parçası bağlam olarak veriliyor) (50.6 sn)

 • ALWAYS_IGNORE_DIRS (coderag/config.py:17):
     ALWAYS_IGNORE_DIRS = {".coderag", ".git", "node_modules", "bin", "obj", "__pycache__", ".venv", ...}
   Bu klasörler her zaman indekslemede atlanıyor.
 • MAX_FILE_BYTES (coderag/config.py:20):
     MAX_FILE_BYTES = 1_000_000

                  Kaynaklar
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Dosya                   ┃ Satırlar ┃ Sembol ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ coderag/core/scanner.py │ 1-10     │        │
│ coderag/cli.py          │ 117-176  │ ask    │
│ coderag/config.py       │ 1-49     │        │
│ coderag/cli.py          │ 99-114   │ index  │
│ coderag/core/scanner.py │ 21-47    │ scan   │
└─────────────────────────┴──────────┴────────┘
```

Bu, bu repo üzerinde yapılmış gerçek bir çalıştırmadır. Belirtilen satırlar doğrudur. Arama adımı,
Sınırlamalar'da anılan bir zayıflığı gösteriyor: çevrilen sorgu test dosyalarına kaymış, doğru dosyalar
yine de bulunmuş.

## Özellikler

- **Yapıya duyarlı parçalama.** Python, tree-sitter ile fonksiyon ve sınıf sınırlarında bölünür. Diğer
  diller (C#, JS/TS, Java, Go, Markdown) LangChain'in dile duyarlı metin bölücüsünü kullanır.
- **Hibrit arama.** Yoğun embedding (`jina-embeddings-v2-base-code`) ile BM25 kelime araması birlikte
  çalışır; `getUserById` ile `get_user_by_id` aynı kelimelere ayrılır. Sonuçlar RRF ile birleştirilir.
- **Diller arası sorular.** Soru, indeksteki gerçek sembol adları kullanılarak İngilizce bir arama
  sorgusuna çevrilir; Türkçe soru İngilizce adlandırılmış kodu bulabilir.
- **Kaynak gösteren cevaplar.** Kod LLM'e gerçek satır numaralarıyla gider, cevap `dosya:satır` belirtir.
- **Artımlı indeksleme.** Dosya hash'leri saklanır, yalnızca değişenler yeniden işlenir. `ask`, cevaplamadan
  önce indeksi kendiliğinden günceller.
- **Akışlı cevap ve görünür ilerleme.** Her adım süresiyle birlikte yazılır.
- **İsteğe bağlı reranker** (`--rerank`) ve **yerleşik değerlendirme düzeneği** (`eval/`).
- **Varsayılan olarak yerel.** Embedding'ler ONNX ile CPU'da, LLM Ollama ile çalışır; vektör deposu
  telemetrisi kapalı yerel bir ChromaDB klasörüdür.

## Kurulum

Python 3.10+ gerekir. Varsayılan kurulum için [Ollama](https://ollama.com) de gerekir.

```bash
git clone https://github.com/meteyilmaz01/code-rag-cli.git
cd code-rag-cli
python -m venv .venv
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
pip install -e ".[ollama]"       # ya da ".[openai]" / ".[gemini]"
ollama pull qwen2.5-coder:7b     # yaklaşık 4,7 GB
```

İlk kullanımda embedding modeli (`jinaai/jina-embeddings-v2-base-code`) Hugging Face'ten `~/.coderag/models`
klasörüne iner. İsteğe bağlı reranker modeli (yaklaşık 1,1 GB) ilk `--rerank` kullanımında iner.

## Kullanım

```bash
coderag index proje/yolu                            # ilk indeks (ya da: coderag index .)
coderag ask "Artımlı indeksleme nasıl çalışıyor?" --path proje/yolu
```

| Seçenek | Anlamı |
|---|---|
| `--path`, `-p` | İndekslenmiş proje dizini (varsayılan `.`) |
| `--show-code`, `-c` | Bulunan kodu satır numaralarıyla da yazdırır |
| `--top-k`, `-k` | LLM'e verilen kod parçası sayısı (varsayılan 5) |
| `--rerank` | Adayları çapraz-kodlayıcıyla yeniden sıralar (daha iyi sıralama, soru başına yaklaşık 20 sn yavaş) |
| `--no-update` | Cevaptan önceki otomatik indeks güncellemesini atlar |

İndeks, indekslenen projenin içindeki `.coderag/` klasöründe durur. `python -m coderag` de çalışır.
CLI mesajları Türkçedir; soruyu LLM'in anladığı herhangi bir dilde sorabilirsin.

### LLM seçimi

| Değişken | Anlamı | Varsayılan |
|---|---|---|
| `CODERAG_PROVIDER` | `ollama`, `openai` veya `gemini` | `ollama` |
| `CODERAG_MODEL` | Model adı | sağlayıcının varsayılanı, bkz. `coderag/config.py` |

Bulut sağlayıcılar için `OPENAI_API_KEY` / `GOOGLE_API_KEY` tanımlı olmalı. **Yalnızca Ollama yolu uçtan uca
denendi**; OpenAI ve Gemini yolları LangChain üzerinden yazıldı ama test edilmedi.

## Nasıl çalışır

```
dosyalar ─▶ tarayıcı ─▶ parçalayıcı ─▶ embedding ─▶ ChromaDB ─┐
(.gitignore'a uyar)  (tree-sitter)   (fastembed)   + BM25     │
                                                              ▼
soru ─▶ İngilizceye çevir ─▶ anlam + kelime araması ─▶ RRF ─▶ (reranker) ─▶ LLM ─▶ kaynaklı cevap
```

| Modül | Görevi |
|---|---|
| `coderag/core/scanner.py` | Klasörü gezer, `.gitignore`'a uyar, ikili ve büyük dosyaları atlar |
| `coderag/core/chunker.py` | Dosyaları satır numaralı fonksiyon/sınıf boyutunda parçalara böler |
| `coderag/core/vector_store.py` | Embedding, ChromaDB, BM25, dosya hash kayıtları |
| `coderag/core/indexer.py` | Tara, parçala, kaydet; artımlı olarak |
| `coderag/core/reranker.py` | İsteğe bağlı çapraz-kodlayıcı puanlama |
| `coderag/core/rag_engine.py` | Sorgu çevirisi, hibrit arama, prompt, LLM çağrısı |
| `coderag/cli.py` | Typer + Rich arayüzü |

## Ölçümler

`eval/run_eval.py`, **bu reponun kendisi** hakkında 16 soru sorar ve cevabı içeren fonksiyonların üst
sonuçlarda çıkıp çıkmadığına bakar. Reranker kapalıyken güncel sonuç:

| Recall@3 | Recall@5 | Recall@8 | MRR |
|---|---|---|---|
| 0,78 | 0,81 | 0,81 | 0,66 |

Yol boyunca sayıları ne değiştirdi (MRR, proje büyürken aynı sette ölçüldü):

| Değişiklik | MRR |
|---|---|
| Hibrit arama başlangıcı | 0,44 |
| Gerçek sembol adlarını gören sorgu çevirisi | 0,68 |
| Test ve doküman parçalarının puanını düşürmek | 0,71 |

Reranker deneyi çoğunlukla olumsuz çıktı: küçük İngilizce bir reranker recall'u düşürdü (0,78'den 0,66'ya),
çok dilli olan sıralamayı iyileştirdi (MRR 0,44'ten 0,54'e) ama yeni bir şey bulamadı ve soru başına yaklaşık
20 sn ekledi. Bu yüzden varsayılan olarak kapalı.

Bu sayıları dikkatle oku: soru seti küçük, yazar tarafından yazıldı ve projenin kendi kodunu kapsıyor.
Genel doğruluğu değil, yönü gösterirler. Farklı aşamalardaki sayılar farklı indeks boyutlarında alındı.

Sadece CPU'lu bir dizüstünde (15,7 GB RAM, ekran kartı yok) `qwen2.5-coder:7b` ile tipik hız: **soru başına
yaklaşık bir dakika**, bunun neredeyse tamamı LLM süresi.

## Sınırlamalar

- **Ekran kartı olmadan yavaş.** Yerel 7B model yanıt süresini belirler. Bulut sağlayıcılar daha hızlı
  olmalı ama denenmedi.
- **Sözdizimine duyarlı parçalama yalnızca Python'da var.** Diğer diller metin bölücü kullanıyor; özellikle
  C# desteği henüz birinci sınıf değil. Çok uzun fonksiyonlar ayrıca bölünmüyor.
- **Sorgu çevirisi fazladan bir LLM çağrısı demek** ve yanılabilir. Örneğin bazen sembol listesindeki test
  fonksiyonu adlarını arama sorgusuna kopyalıyor.
- **Küçük ve yazar tarafından yazılmış değerlendirme.** Herkese açık benchmark'larla karşılaştırma yok.
- **Cevaplar yüzeysel kalabilir.** Model bazen doğru değerleri verip mekanizmayı anlatmayabiliyor.
- **Yalnızca Windows 11 ve Python 3.14'te denendi.** Python 3.10+ bildirildi, ama eski sürümler ve
  Linux/macOS çalıştırılmadı. Temiz ortamda kurulum CI ile doğrulanmadı.
- **Ağır bağımlılıklar** (ChromaDB, ONNX Runtime, tree-sitter dil paketi) ve ilk çalıştırmada büyük model indirmeleri.

## Geliştirme

```bash
pip install -e ".[dev,ollama]"
pytest                          # 57 test, model indirmeden çalışır
python eval/run_eval.py --only 0   # arama kalitesi; --only ile yapılandırma seçilir
```

## Lisans

[MIT](LICENSE)
