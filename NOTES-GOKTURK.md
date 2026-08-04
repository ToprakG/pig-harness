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
