"""Bağlam sıkıştırma (compaction) — atılan mesajları silmeden önce damıt.

NEDEN
-----
OpenAI'nin ARC-AGI-3 ölçümü (2026-07, "how two settings tripled our ARC-AGI-3
scores"): resmi harness her adımdan sonra modelin akıl yürütmesini atıyor ve
bağlam dolunca eski aksiyonları kesiyordu. Bu iki ayarı düzeltmek —
**retained reasoning** ve **compaction** — GPT-5.6 Sol'un public set skorunu
%13.3 → %38.3 çıkardı ve çıktı token'ını 6 kat azalttı.

Duck harness bu ikisinden BİRİNİ zaten doğru yapıyor: `assistant_message`
içine `reasoning` konuyor, yani akıl yürütme adımlar arası korunuyor.
Eksik olan ikincisi: `_drop_oldest_history_block` bağlam dolunca en eski
bloğu **hiçbir iz bırakmadan siliyor**. Model o turda ne denediğini, neyin
işe yaramadığını unutuyor ve aynı hamleyi tekrar ediyor — ARC-AGI-3'te her
tekrar edilen hamle puanı KARESEL olarak düşürüyor.

NASIL
-----
Atılan bloklardan LLM çağrısı yapmadan, deterministik olarak damıtıyoruz.
Ekstra gecikme ve token maliyeti yok — 9 saatlik duvar saati bütçesinde bu
önemli. Çıkarılan bilgi, aksiyon verimliliği için en değerli olanı:

  * hangi aksiyonlar yürütüldü (sırayla),
  * hangileri tahtayı DEĞİŞTİRMEDİ (tekrar edilmemesi gereken hamleler),
  * seviye geçişleri ve game over'lar,
  * modelin kendi kısa bulguları.

Sonuç, sistem mesajından hemen sonra kalıcı bir "COMPACTED HISTORY" bloğu
olarak bağlamda kalır.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

MAX_NOTES = 40
MAX_NOTE_CHARS = 220
MAX_BLOCK_CHARS = 2000

_ACTION_RE = re.compile(
    r"\b(UP|DOWN|LEFT|RIGHT|SPACE|MOUSE|ACTION[1-7]|RESET)\b")
_NOCHANGE_RE = re.compile(
    r"board_changed[\"']?\s*[:=]\s*(false|False)|no change|did not change",
    re.IGNORECASE)
_LEVEL_RE = re.compile(
    r"level_completed[\"']?\s*[:=]\s*(true|True)|level\s+(\d+)", re.IGNORECASE)
_GAMEOVER_RE = re.compile(r"game_over[\"']?\s*[:=]\s*(true|True)", re.IGNORECASE)


def _text_of(message: dict[str, Any]) -> str:
    """Mesajdan düz metin çıkar (content + reasoning + tool_calls)."""
    parts: list[str] = []
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict):
            args = function.get("arguments")
            if isinstance(args, str):
                parts.append(args)
            elif isinstance(args, dict):
                parts.append(json.dumps(args))
    return "\n".join(parts)[:MAX_BLOCK_CHARS]


def summarize_dropped(messages: list[dict[str, Any]]) -> list[str]:
    """Atılacak mesaj bloğundan kalıcı notlar üret.

    LLM çağrısı YOK — tamamen deterministik. Ekstra gecikme/token maliyeti
    olmadan, verimlilik için en değerli bilgiyi tutar.
    """
    if not messages:
        return []

    actions: list[str] = []
    inert: list[str] = []
    events: list[str] = []
    findings: list[str] = []

    # Etkisiz hamlenin ADI onceki assistant mesajinda, "degismedi" bilgisi ise
    # tool mesajinda duruyor. Ikisini iliskilendirmezsek notumuz
    # "(an action) produced no change" gibi ise yaramaz bir sey oluyor --
    # oysa tekrar edilmemesi gereken HAMLE ADI en degerli bilgi.
    pending: list[str] = []

    for message in messages:
        role = str(message.get("role", "")).strip()
        text = _text_of(message)
        if not text:
            continue

        found = _ACTION_RE.findall(text)
        if role == "assistant" and found:
            pending = list(dict.fromkeys(found))
            for name in found:
                if not actions or actions[-1] != name:
                    actions.append(name)

        if role == "tool":
            if _NOCHANGE_RE.search(text):
                inert.extend(pending or found[:2] or ["(an action)"])
            pending = []
            if _GAMEOVER_RE.search(text):
                events.append("GAME_OVER occurred")
            level = _LEVEL_RE.search(text)
            if level and level.group(1):
                events.append("a level was completed")

        if role == "assistant":
            for line in text.splitlines():
                line = line.strip()
                if 20 <= len(line) <= MAX_NOTE_CHARS and any(
                        k in line.lower() for k in
                        ("goal", "seems", "appears", "hypoth", "rule",
                         "target", "avatar", "wall", "button", "blocked")):
                    findings.append(line)

    notes: list[str] = []
    if actions:
        notes.append("actions taken (oldest first): "
                     + ", ".join(actions[:24])[:MAX_NOTE_CHARS])
    if inert:
        uniq = list(dict.fromkeys(inert))[:10]
        notes.append("produced NO board change (do not repeat blindly): "
                     + ", ".join(uniq))
    for event in dict.fromkeys(events):
        notes.append(event)
    for finding in list(dict.fromkeys(findings))[:4]:
        notes.append("earlier note: " + finding[:MAX_NOTE_CHARS])
    return notes


class CompactionStore:
    """Sıkıştırılmış notların kalıcı deposu.

    Bir seviye tamamlandığında temizlenir: yeni seviyenin düzeni ve kuralları
    farklı olabilir, eski notlar yanıltıcı hale gelir (duck'ın kendi
    `_update_summarized_knowledge_from_step_summary` mantığıyla aynı gerekçe).
    """

    def __init__(self, max_notes: int = MAX_NOTES) -> None:
        # A/B icin anahtar: PIG_COMPACTION=0 ile kapatilinca sinif tamamen
        # pasif olur ve davranis duck taban cizgisiyle birebir ayni kalir.
        self.enabled = os.environ.get("PIG_COMPACTION", "1").strip() != "0"
        self._notes: list[str] = []
        self._max_notes = max_notes
        self.compactions = 0
        self.dropped_messages = 0

    def add_dropped(self, messages: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        notes = summarize_dropped(messages)
        if not notes:
            return
        self.compactions += 1
        self.dropped_messages += len(messages)
        for note in notes:
            if note not in self._notes:
                self._notes.append(note)
        if len(self._notes) > self._max_notes:
            # En eskileri at, ama "no board change" notlarını koru: tekrar
            # edilen etkisiz hamleler puanın en büyük sızıntısı.
            keep = [n for n in self._notes if "NO board change" in n]
            rest = [n for n in self._notes if "NO board change" not in n]
            self._notes = (keep + rest)[-self._max_notes:]

    def reset(self) -> None:
        self._notes.clear()

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    def as_lines(self) -> list[str]:
        if not self.enabled or not self._notes:
            return []
        return [
            "COMPACTED HISTORY (older turns were compressed, not discarded):",
            *[f"- {note}" for note in self._notes],
            "- Treat these as evidence from earlier in this level, "
            "not as current board state.",
        ]

    def as_message(self) -> dict[str, Any] | None:
        lines = self.as_lines()
        if not lines:
            return None
        return {"role": "user", "content": "\n".join(lines)}
