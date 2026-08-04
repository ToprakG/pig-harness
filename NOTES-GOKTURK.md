# Göktürk şubesi — değişiklik günlüğü

Temel: Tufa Labs duck harness (public set ortalaması **1.60**, medyan 0.07,
500 koşuda 0 kazanma). Buradaki değişiklikler ölçülmüş gerekçelere dayanıyor.

## 1. Compaction — `inference/agent/compaction.py` (YENİ)

**Gerekçe.** OpenAI'nin ARC-AGI-3 ölçümü (2026-07): resmi harness her adımdan
sonra modelin akıl yürütmesini atıyor ve bağlam dolunca eski aksiyonları
kesiyordu. İki ayarı düzeltmek — *retained reasoning* + *compaction* —
GPT-5.6 Sol'un public set skorunu **%13.3 → %38.3** çıkardı ve çıktı token'ını
**6 kat** azalttı.

Duck bunlardan birini zaten doğru yapıyor: `assistant_message["reasoning"]`
ile akıl yürütme adımlar arası korunuyor. **Eksik olan compaction'dı:**
`_drop_oldest_history_block` bağlam dolunca en eski bloğu hiçbir iz
bırakmadan siliyordu. Model ne denediğini ve neyin işe yaramadığını unutup
aynı hamleyi tekrar ediyor — ARC-AGI-3'te her tekrar puanı karesel düşürüyor.

**Uygulama.** Atılan bloklar silinmeden önce deterministik olarak damıtılıyor
(LLM çağrısı yok → ek gecikme ve token maliyeti yok, 9 saatlik bütçede önemli):
- yürütülen aksiyonlar, sırayla
- **tahtayı değiştirmeyen hamleler** (aksiyon adıyla birlikte)
- seviye geçişleri ve GAME_OVER olayları
- modelin kendi kısa bulguları

Notlar sistem mesajından sonra kalıcı `COMPACTED HISTORY` bloğu olarak duruyor.
Seviye geçişinde temizleniyor (yeni seviyenin kuralları farklı olabilir —
duck'ın kendi world-model sıfırlama mantığıyla aynı gerekçe).

Bir ayrıntı önemliydi: etkisiz hamlenin *adı* assistant mesajında, "değişmedi"
bilgisi tool mesajında duruyor. İlişkilendirmeden not `"(an action) produced
no change"` gibi işe yaramaz çıkıyordu; şimdi `"MOUSE"` diyor.

## 2. Verimlilik çerçevesi — `inference/agent/prompts.py`

Duck'ın prompt'u sadece *"as few actions as possible"* diyordu. Puanlamanın
karesel olduğunu ve insan baseline'ının ne olduğunu söylemiyordu. Eklendi:
- `level_score = (human_actions / your_actions)^2`; 2 kat = puanın çeyreği
- insan L1 baseline'ı ~30 aksiyon (public sette 7-78)
- etkisiz hamleyi tekrarlamak en büyük puan sızıntısı

Bu çerçeveyi kendi harness'ımızda ölçtük: L1 medyan oranı **3.74x → 0.83x**,
skor **0.07 → 0.30** (skor tablosu ölçeği). Bkz. `../Brenchmark/NOTES.md`.

## Ölçüm notu

Skor tablosu ölçeği 0-100'dür: `min(115, (baseline/actions)^2 * 100)`.
Referans noktaları: duck 1.60 · lider 1.86 · *L1'i 25 oyunun hepsinde 1.0x
ile geçmek 3.52*. Saha tamamen "derine in, verimliliği umursama" bölgesinde;
verimlilik hâlâ açık bir cephe.

## Henüz yapılmadı
- Bu değişikliklerin duck taban çizgisine karşı A/B ölçümü (asıl iş).
- Agentic Harness sözleşmesine paketleme (`agent/my_agent.py`,
  `agent/harbor_agent.py`, `main.py`) — bu repo şu an Kaggle notebook şeklinde.

## A/B düzeneği

`run_ab.sh` — aynı kod, tek fark `PIG_COMPACTION` (0/1). `CompactionStore`
kapalıyken tamamen pasif, yani kapalı kol duck taban çizgisiyle birebir aynı.

**Neden az oyun, çok pass:** iki kol arasındaki farkı görünür kılan şey oyun
çeşitliliği değil, aynı oyunda tekrarlanan koşuların varyansının ortalanması.
Duck'ın kendi sayıları bunun ne kadar gürültülü olduğunu gösteriyor —
ar25'in 20 pass skoru `0.0, 0.0, 0.0, 0.04, ... 6.81, 7.06, 8.33`. Tek pass'a
bakıp karar vermek yanıltıcı olur.

Seçilen oyunlar: **ft09** ve **vc33** — duck'ın ısı haritasında en tutarlı
sıfır-olmayan iki oyun (ft09 ort. 10.28, vc33 ort. 3.33). Sıfır alan oyunlarda
A/B hiçbir şey ölçemez.

## Yerelde duck'ı koşturma (kurulum notları)

vLLM gerekmiyor — o sadece opsiyonel `server` ekstrası. Çekirdek bağımlılıklar
hafif (arcengine, matplotlib, python-dotenv, requests, taaf).

Karşılaşılan üç engel ve çözümü:
1. `requires-python = "==3.12.12"` çok katı → `>=3.12,<3.13`.
2. Editable kurulumun `.pth` dosyası yüklenmiyor (dizin adında **boşluk** var:
   `Exposure AI`). Çözüm: `PYTHONPATH` ile hem `ARC3-Inference` hem
   `tufa-arc-agi-framework/src` verilmeli.
3. `base_url` config'den değil `LOCAL_ANALYZER_BASE_URL` / `OPENAI_BASE_URL`
   env değişkeninden okunuyor; `OPENAI_PROVIDER=cerebras` ile birlikte
   verilmezse yerel vLLM'e (127.0.0.1:1234) gidip sessizce 0 aksiyon üretiyor.

Çalışan komut `run_ab.sh` içinde.

## A/B sonuç #1 (ft09 + vc33, 3 pass, 10 dk/oyun) — SONUÇSUZ

| kol | ortalama | medyan | aksiyon | token | compaction |
|---|---|---|---|---|---|
| off | 0.00 | 0.00 | 1204 | 464.372 | 0 kez |
| on  | 0.60 | 0.00 | 1287 | 465.022 | **240 kez** |

Mekanizma çalıştı: 240 tetikleme, 6 transkriptin hepsinde `COMPACTED HISTORY`.
Kapalı kolda hiç tetiklenmedi. Düzenek doğru.

**Ama sonuç istatistiksel olarak anlamsız.** Farkın tamamı tek bir pass'tan
geliyor (vc33 bir pass'ta 1.19, geri kalan 5 koşu her iki kolda da 0). Medyan
iki kolda da 0.00. 6 koşuda 1-0 farkı → Fisher p≈0.5.

Duck'ın kendi verisi bu tuzağı zaten gösteriyordu: ar25'in 20 pass skoru
`0.0, 0.0, 0.0, 0.04, ... 6.81, 7.06, 8.33`. Bu dağılımda 3 pass ile karar
vermek kumar.

Zayıf ama yönü doğru bir yan gözlem: "on" kolu aynı token bütçesiyle daha
fazla aksiyon aldı (1287 vs 1204) — token başına daha az tekrar. Compaction
ek token maliyeti getirmedi (deterministik olduğu için beklendiği gibi).

Not: mutlak skorlar duck'ın yayınladığından çok düşük (ft09 için onlar 10.28
diyor, biz 0.00). İki sebep: 10 dk/oyun bütçe (onlar 45 dk) ve farklı model
(gpt-oss-120b vs Qwen3.6-27B-FP8). Bu A/B mutlak skoru değil, iki kol
arasındaki FARKI ölçmek için kuruldu.
