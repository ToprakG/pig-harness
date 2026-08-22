# Göktürk V7: animasyon-farkindalikli board_changed raporlama

## Bulgu (kaynak)

Rakip Kaggle notebook `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`,
`feature/animation-awareness` branch. Kendi `noop_guard.py` + `animation.py`
modulleri, arcengine'in bazi aksiyonlar icin coklu kare (animasyon) dondugunu
ve ilk surumlerinin bunu yanlis isledigini belgeliyor: settled (son) kare
oncekiyle ayni gorunse bile (board_changed=False), aksiyonun ara karelerde
gercek bir etkisi olmus olabilir -- reddedilen tiklama, tuketilen deneme, vb.
Onlarin duzeltmesi: `board_changed or animated` -- animasyonlu bir aksiyon
ASLA "kesin etkisiz" sayilmiyor.

Ayni desen `feature/animation-awareness`'ta `ft09`/`sb26` gibi cok kareli
oyunlarda calisan aksiyonlarin yanlislikla olu/etkisiz damgalanmasina
sebep oluyordu.

## Bizim durumumuz: pig-harness'ta bu bug NASIL farkli tezahur ediyor

Kendi `agent/arc3` harness'imizda (Faz 4b, ayni bulgu) bu bir OTOMATIK
filtre bug'iydi: `DeadSignatureStore` no-op'lari aday listesinden siliyor,
animasyon-korumasi olmadan calisan tiklamalari kaybediyorduk.

Pig-harness/duck mimarisi FARKLI: burada boyle bir otomatik filtre YOK.
`board_changed` sadece LLM'e metin/JSON olarak bilgi olarak veriliyor
(`tool_agent.py::_compact_action_result`, `_describe_last_outcome`,
`programmatic_memory.py::append_action`). Yani "olu" olarak silinen bir
aday listesi yok -- ama YANLIS bilgi LLM'in kendi akil yurutmesini
yanlis yonlendirebilir: `board_changed=False` gorduginde LLM aksiyonun
hicbir etkisi olmadigini varsayip onu terk edebilir, halbuki settled
karede geri alinmis gercek bir etki olabilir.

`taaf/game.py::GameState` zaten coklu kare bilgisini tasiyor
(`raw.frame`: tum kareler, `.frame`: son/settled kare, `.intermediate_frames`:
`raw.frame[:-1]`) ama `solver.py::_execute_action` sadece settled karesi
karsilastiriyordu (`board_changed = previous_grid != _grid_from_state(new_state)`),
ara kare bilgisini tamamen atiyordu.

## Degisiklik

`animated = len(new_state.raw.frame) > 1` hesaplanip asagidaki noktalara
yeni bir `animated` alani olarak eklendi (OTOMATIK FILTRE DEGIL, sadece
LLM'e dogru bilgi):

- `solver.py::_execute_action` payload (`"animated": animated`)
- `solver.py::execute_batch` toplu payload (`any(...)`)
- `solver.py` -> `record_environment_action(..., animated=animated)`
- `programmatic_memory.py::append_action` -- log metnine, SADECE
  `animated and not board_changed` durumunda uyari satiri ekleniyor
  ("do not treat board_changed=False here as proof the action was inert")
- `tool_agent.py::_compact_action_result` -- LLM'in dogrudan gordugu
  aksiyon sonucu JSON'una `animated` alani eklendi
- `tool_agent.py::_summarize_executed_results` -- batch ozeti icin `animated`
- `tool_agent.py::_describe_last_outcome` -- board_changed=False VE
  animated=True durumunda metin degisti: eskiden duz "did not show a
  confirmed board change; treat this as weak evidence" diyordu (yanlis
  yonlendirici), simdi "showed a multi-frame (animated) response ...
  do NOT treat this as proof the action was inert" diyor.

`board_changed`'in kendi hesaplanma mantigi DEGISMEDI (hala dogru: settled
kare farki). Sadece yaninda dogru baglami tasiyan yeni bir alan eklendi.

## Dogrulama

- Uc dosya da `ast.parse` ile sozdizimi kontrolunden gecti.
- `animated` mantigi (`len(raw.frame) > 1`), `taaf/game.py` kaynagindan
  dogrudan okunarak dogrulandi (satir 168-180: `frame` property son
  karayi, `intermediate_frames`/`all_frames` property'leri ara/tum
  kareleri donuyor -- `raw.frame` listesi zaten coklu kare tasiyordu,
  sadece kullanilmiyordu).
- arcengine paketi bu ortamda kurulu olmadigindan (`ModuleNotFoundError`)
  uctan uca calisan bir entegrasyon testi YAPILAMADI -- bu, mantigin
  yanlis oldugu anlamina gelmez, sadece bu ortamda calistirilamadi demek.

## Bu ne SOYLEMIYOR

- Bu degisiklik LLM'in davranisini garanti degistirmez -- sadece ona
  daha dogru bilgi sunar. LLM yine de yanlis karar verebilir.
- Skor artisi VAAT ETMIYORUZ. Bu, V6'nin wa30 hipotezi gibi, olcumle
  degil kod-okumasi + mantik yurutmesiyle bulunan bir dogruluk duzeltmesi.
- Kendi `agent/arc3` harness'imizdaki gibi bir "otomatik dead-signature
  filtresi" pig-harness'a EKLENMEDI -- boyle bir mekanizma orada zaten
  yok, ve bu fix onu eklemiyor, sadece var olan bilgi akisini duzeltiyor.
