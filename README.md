# code-rag-cli

Yerel bir kod tabanını indeksleyip terminalden soru sormayı sağlayan Code-RAG aracı.
Yanıtlar ilgili dosya ve satır numaralarıyla birlikte gelir.

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[ollama]"      # ya da ".[openai]" / ".[gemini]"
```

## Kullanım

```
coderag index <dizin>                       # kodu tara ve indeksle (tekrarda sadece değişenler)
coderag ask "<soru>" --path <dizin>         # soru sor
coderag ask "<soru>" --path <dizin> -c      # kaynak kodu da göster
coderag ask "<soru>" --top-k 8              # LLM'e verilen kod parçası sayısı (varsayılan 5)
coderag ask "<soru>" --rerank               # reranker ile yeniden sırala (daha iyi sıralama, soru başına ~20 sn ek süre)
```

İndeks, indekslenen dizinin içindeki `.coderag/` klasörüne yazılır. `ask` her soru öncesi indeksi değişen
dosyalara göre kendiliğinden günceller (`--no-update` ile atlanır); ilk indeksi `coderag index` ile oluşturursun.

## LLM seçimi

Ortam değişkenleriyle ayarlanır:

| Değişken | Anlamı | Varsayılan |
|---|---|---|
| `CODERAG_PROVIDER` | `ollama`, `openai` veya `gemini` | `ollama` |
| `CODERAG_MODEL` | model adı | sağlayıcıya göre (bkz. `config.py`) |

OpenAI için `OPENAI_API_KEY`, Gemini için `GOOGLE_API_KEY` tanımlı olmalı.

Embedding modeli (`jinaai/jina-embeddings-v2-base-code`) ilk `index` çalıştırmasında indirilir.
Reranker varsayılan olarak kapalıdır; `--rerank` ile ilk kullanımda `jinaai/jina-reranker-v2-base-multilingual`
(yaklaşık 1,1 GB) indirilir.

## Arama kalitesini ölçme

```
python eval/run_eval.py
```

`eval/questions.json` içindeki sorular için beklenen fonksiyonların ilk 3, 5 ve 8 sonuçta çıkıp çıkmadığını
(Recall@k) ve doğru cevabın ortalama sırasını (MRR), reranker'lı ve reranker'sız yapılandırmalar için karşılaştırır.

## Nasıl çalışır

1. `core/scanner.py`: `.gitignore`'a saygı duyarak kod dosyalarını toplar.
2. `core/chunker.py`: Python'u tree-sitter ile fonksiyon/sınıf sınırlarında, diğer dilleri LangChain splitter ile böler.
3. `core/vector_store.py`: fastembed ile embedding üretir, ChromaDB'ye yazar; ayrıca BM25 kelime araması sunar.
4. `core/rag_engine.py`: anlam ve kelime aramasını RRF ile birleştirir, `core/reranker.py` ile yeniden sıralar,
   bulunan kodu LangChain üzerinden LLM'e verir.
5. `cli.py`: Typer + Rich arayüzü.

## Test

```
pip install pytest
pytest
```
