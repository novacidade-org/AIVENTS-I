import langid
import re
from unidecode import unidecode
from langchain.memory import ChatMessageHistory
from constants import selected_tables, event_triggers, event_table
from config import Config

def detect_lang(text, prefix_words=6):
    prefix = " ".join(text.split()[:prefix_words])
    lang, _ = langid.classify(prefix)
    return 'pt' if lang == 'pt' else 'en'

# function to check if the user question regards some event
def looks_like_event_q(text: str) -> bool:
    t = unidecode(text or "").lower()
    return any(w in t for w in event_triggers)

def boost_events_first(cands: list[str], top_k: int) -> list[str]:
    # keep order but force events table to the front once
    out = [event_table] + [t for t in cands if t != event_table]
    # respect the global allowlist just in case
    out = [t for t in out if t in selected_tables]
    return out[:top_k]

def make_join_hints(config = Config(), tables: list[str] = selected_tables, max_per_table: int = 4) -> str:
        """
        Build a short, token-safe list of join hints from the glossary relations
        for the provided tables. One line per relation, capped per table.
        """
        tm = config.table_meta
        # tm = getattr(config, "table_meta", {})  # contains 'relations' per table (normalized)
        lines = []
        for t in tables:
            rels = (tm.get(t, {}) or {}).get("relations", []) or []
            shown = 0
            for r in rels:
                to_table = r.get("to_table")
                if not to_table or to_table not in tables:
                    continue  # skip relations pointing outside the current scope
                from_col = r.get("from_col")
                to_col   = r.get("to_col")
                kind     = r.get("kind", "fk")
                if from_col and to_col:
                    # example: silver_fact...fk_concelho = silver_dim...fk_concelho (fk)
                    lines.append(f"- {t}.{from_col} = {to_table}.{to_col} ({kind})")
                    shown += 1
                    if shown >= max_per_table:
                        break
        return "\n".join(lines) if lines else "(none)"

# Memory
chat_store = {}
long_term_memory = {}

def get_chat_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

def get_long_term_memory(session_id: str):
    return ". ".join(long_term_memory.get(session_id, []))

def is_select(sql: str) -> bool:
    """True if the statement looks like a SELECT."""
    return bool(re.match(r"\s*select\b", (sql or ""), re.I))

def looks_like_probe(sql: str) -> bool:
    """
    Returns True for classic probe-ish queries:
    - TOP 1 / LIMIT 1
    - EXISTS(...)
    - SELECT 1 ...
    Everything else is NOT a probe (even plain SELECT without aggregates).
    """
    if not sql:
        return True
    
    # normalize whitespace and add spaces at ends for simple ' in ' contains
    s  = " " + " ".join(sql.split()) + " "
    sl = s.lower()

    if re.search(r"\bselect\s+top\s+\d+\b", sl): return True
    if re.search(r"\blimit\s+\d+\b", sl): return True
    if " exists " in sl: return True
    if re.search(r"\bselect\s+1\b", sl): return True
    # Don’t force aggregates. A plain SELECT listing rows is valid and not a probe.
    return False