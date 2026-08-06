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

## A/B #2 GEÇERSİZ — kök neden: zaman aşımı çöküşü

20 koşuluk "off" kolu 3 sa 8 dk sürdü ve **hepsi 0** aldı. Transkriptlerde
**5.953 istek hatası**, bunların 5.788'i:

```
request_error: HTTPSConnectionPool(host='api.cerebras.ai', port=443):
Read timed out. (read timeout=0.1)
```

`inference/framework/solver.py:227` `request_timeout_seconds()` istek zaman
aşımını şu üçünün **minimumu** olarak alıyor: yapılandırılmış timeout,
oyunun kalan süresi, soft kalan süre. `max_runtime_minutes=10` ile sınıra
yaklaşıldığında timeout 0.1 saniyeye çöküyor ve sonraki her istek anında
başarısız oluyor. Ajan döngüde kalıp bütçeyi ölü isteklerle yakıyor.

Sonuç: koşu başına medyan **35 aksiyon** (ft09'un L1 insan baseline'ı 43).
Ajan seviye 1'i bitirmeye fiziksel olarak yetişemiyor → skor her koşuda 0
garantili. **Bu düzenekte compaction farkı ölçülemez.**

Duck'ın kendi varsayılanı `max_runtime_minutes: 45`. Benim 10 dakikam,
üretken pencereyi ölü zamanın altında bıraktı. A/B #1 de (10 dk) aynı
sebeple şüpheli — oradaki tek 1.19'luk pass muhtemelen gürültü.

**Düzeltme:** süreyi 30-45 dk'ya çıkar, eşzamanlılığı düşür (her koşu daha
çok istek payı alsın), pass sayısını azalt. Uzun koşu zaten şart: compaction
ancak bağlam dolunca devreye giriyor.

## Zaman aşımı çöküşü — gerçek kök neden ve düzeltme

Yukarıdaki #2 teşhisi eksikti. "Oyunun kendi süre sınırına yaklaşınca timeout
çöküyor" açıklaması mekanizmayı doğru anlatmıyor, çünkü oyun-başı kalan süre
**kendi kendini sınırlıyor**: `remaining` sıfıra ulaştığı an
`runtime_limit_reached()` true oluyor ve `should_stop()` döngüyü zaten
kesiyor. Bu terim en fazla tek bir ölü istek üretebilir.

Süresiz dönmeye yol açan terim `soft_remaining`. `soft_end_time` **oturum
geneli** bir işarettir, oyun başına değil — `taaf/deploy_inline.py:110`:

```python
_buffer = min(_SOFT_DEADLINE_BUFFER_S, _budget / 2)
soft_end_time = datetime.now() + timedelta(seconds=_budget - _buffer)
```

### Doğrulanan (kod + birincil kaynak)

`soft_end_time` formülü `ab-baseline` artefaktlarında birebir üretiliyor:

| kaynak | değer |
|---|---|
| `deploy_meta.json` → `target_config.max_runtime_s` | 2400 s |
| `deploy_meta.json` → `started_at` | 00:58:00.697 |
| hesap: `_buffer = min(600, 2400/2)` = 600 → `started_at + 1800 s` | 01:28:00.697 |
| `stdout.log` → `deploy.inline: soft_end_time` | 01:28:00.698 |

Buradan çıkan mekanizma, tamamen kod okumasıyla:

1. `soft_time_remaining_seconds()` soft deadline geçtikten sonra kalıcı
   olarak **0.0** döner (`max(0.0, ...)`).
2. `request_timeout_seconds()` bunu `min()` içine katıyor ve sonucu
   `max(0.1, ...)` ile tabanlıyor → istek timeout'u **0.1 s**.
3. `should_stop()` içinde soft deadline kontrolü **yok** → döngü bitmiyor,
   1 s'lik retry backoff ile saniyede ~1 ölü istek.

Oyun-başı terim bu soruna yol açamaz: `remaining` sıfıra ulaştığı an
`runtime_limit_reached()` true olur ve `should_stop()` keser. Yani sınırsız
dönmeyi mümkün kılan **tek** terim `soft_remaining`.

### Çıkarım (doğrulanamadı)

A/B #2'deki 5.788 `read timeout=0.1` hatasının bu mekanizmadan geldiği
**çıkarımdır, ölçüm değildir.** O koşunun logları `/tmp/pig-ab` altındaydı
ve silinmiş; geri getirilemiyor. Elimizdeki koşularda (`ab-baseline` dahil)
`timeout=0.1` **hiç geçmiyor** — bu mekanizmayla çelişmiyor (tek dalga,
soft deadline sonrası başlayan oyun yok) ama onu kanıtlamıyor da.

Ayrıca dikkat: `run_ab.sh` `--max-runtime-minutes 10` veriyor, bu **oyun
başı** sınırı ayarlıyor; oturum `max_runtime_s`'i ayrı bir yoldan geliyor
(`ab-baseline`'da 1800 s'e karşı 2400 s). A/B #2 için o iki sayının ne
olduğu bilinmiyor, dolayısıyla "ikinci dalga tam soft deadline dolarken
başlıyor" aritmetiği o koşu için gösterilmiş değil.

**Düzeltme** (`inference/framework/solver.py`): `soft_remaining` aday listesinden
çıkarıldı. TAAF'ın kendi sözleşmesi bunu zaten söylüyor (`taaf/solver.py:42`):

> `soft_end_time`: indicative pacing hint. Cancellation is enforced by
> `Benchmark` via `task.cancel()` — solvers never need to check the clock
> themselves.

Yani oturum düzeyinde bir tempo işareti, ağ isteği bütçesi olarak
kullanılmamalıydı.

`should_stop()`'a bilerek soft deadline kontrolü **eklenmedi**: bu, oyunların
ne zaman bittiğini değiştirir, dolayısıyla skorları değiştirir — hata
düzeltmesi kılığında davranış değişikliği olur ve tam da geçerli kılmaya
çalıştığımız A/B'yi kirletir.

Doğrulama durumu: yerelde çalıştırılamadı. Gerçek sebep madde 2'de yazdığım
`.pth` sorunu değil: proje ağacı **iCloud tarafından tahliye edilmiş**
(`ls -lO` → `dataless`). Venv'in `site-packages` altında 616 dataless `.py`
var; iCloud daemon materialize edemiyor (`brctl download` başarı dönüyor ama
dosya dataless kalıyor, `brctl status` asılıyor). Bu yüzden her import,
`find`, hatta `git commit` takılıyor. Kaynak dosyaların 27'si de temiz —
sorun sadece venv ve bazı git scratch dosyalarında.

Çözüm: projeyi iCloud senkronlu Desktop dışına taşı ve venv'i yeniden kur.
Bu aynı zamanda boşluklu dizin sorununu da bitirir (iki engel, tek hamle):

```bash
mv "/Users/gokturkakman/Desktop/Exposure AI/pig-harness" ~/dev/pig-harness
```

Regresyon testi bu taşıma sonrası eklenmeli.

**Bu düzeltme olmadan hiçbir A/B geçerli değil.** Sıradaki koşu, boyutu da
düzeltilmeden tekrarlanmamalı: 2 oyun × 3 pass gürültü bandının çok altında
(A/B #1 tam olarak bu yüzden sonuçsuz kaldı). Bir sonraki koşu ya açıkça
"smoke test" etiketiyle yapılmalı, ya da anlamlı bir n ile.
