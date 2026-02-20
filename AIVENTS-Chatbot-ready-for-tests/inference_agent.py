import re
import json
import langid
import nltk
import spacy
from nltk.corpus import stopwords
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from typing import List, Optional, Any, Iterable, Dict, Tuple
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from config import Config
from constants import event_table, selected_tables, event_triggers
from logger import logger
from utils import detect_lang, looks_like_event_q, boost_events_first, make_join_hints, is_select, looks_like_probe

# detect language
langid.set_languages(['en','pt'])

# Extract lowercase letter-only tokens from text (Unicode-aware).
def tokens_letters(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

# Lemmatize English tokens (noun→verb) while removing English stopwords.
def lemma_stop_en(tokens: Iterable[str]) -> List[str]:
    out = []
    for t in tokens:
        if t in stop_words_en:
            continue
        l = lemma_en.lemmatize(t)          # noun/default
        l = lemma_en.lemmatize(l, pos="v") # verb
        if l and l not in stop_words_en:
            out.append(l)
    return out

# Lemmatize Portuguese tokens with stanza, filtering Portuguese stopwords.
def lemma_stop_pt(tokens: Iterable[str], nlp_pt) -> List[str]:
    # if lemma_pt is None:
    #     raise ValueError("lemma_pt pipeline is required for Portuguese lemmatization.")
    doc = nlp_pt(" ".join(t for t in tokens if t))
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        lem = tok.lemma_.lower()
        if lem and lem not in stop_words_pt and _WORD_RE.fullmatch(lem):
            out.append(lem)
    return out

# Keeps only letters (incl. Portuguese accents), no digits/punct/underscores
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# STOP WORDS - avoid downloads at runtime
nltk_ready = False
def ensure_nltk():
    global nltk_ready
    if nltk_ready: return
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    nltk_ready = True

spacy_pt = None
def get_spacy_pt():
    global spacy_pt
    if spacy_pt is None:
        try:
            spacy_pt = spacy.load("pt_core_news_md")
        except OSError:
            spacy_pt = spacy.blank("pt")  # graceful fallback
    return spacy_pt

# --- minimal compatibility init (adds the old globals) ---
ensure_nltk()
try:
    stop_words_en = set(stopwords.words('english'))
except Exception:
    stop_words_en = set()

try:
    stop_words_pt = set(stopwords.words('portuguese'))
except Exception:
    stop_words_pt = set()

lemma_en = WordNetLemmatizer()  # local, no downloads
nlp_pt = get_spacy_pt()

# Clean a single text: lower → letters-only → stopword removal → lemmatization → (optional) dedupe; returns lemmas.
def clean_text(
    text: str,
    lang: str,                       # "en" or "pt"
    nlp_pt=None,                  # required if lang == "pt"
    extra_stop: Optional[Iterable[str]] = None,
    dedupe: bool = True     # remove repeated lemmas
) -> List[str]:
    toks = tokens_letters(text)
    extra = set(x.lower() for x in (extra_stop or []))

    if lang == "pt":
        # toks = tokens_letters(text)                 
        lemmas = lemma_stop_pt(toks, nlp_pt)         
        lemmas = [w for w in lemmas if w not in extra]
    else:
        # toks = tokens_letters(text)
        lemmas = lemma_stop_en(toks)
        lemmas = [w for w in lemmas if w not in extra]

    lemmas = [unidecode(w) for w in lemmas]

    if dedupe:
        seen, deduped = set(), []
        for w in lemmas:
            if w not in seen:
                seen.add(w)
                deduped.append(w)
        return deduped
    
    return lemmas

def pick_final_query(
    executed_sql: List[Tuple[str, Any]],
    output_text: str,
    run_query_callable,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Decide the *final* query:
      1) Prefer the last executed SELECT that is NOT a probe.
      2) Else, the last executed SELECT of any kind.
      3) Else, parse the last fenced ```sql block, and if it’s a SELECT:
         - Prefer non-probe; otherwise accept any SELECT
         - Execute it now via run_query_callable and return (sql, observation).
    """
    # 1) last executed SELECT that is not a probe
    selects    = [(q, o) for (q, o) in executed_sql if is_select(q)]
    non_probes = [(q, o) for (q, o) in selects if not looks_like_probe(q)]
    if non_probes:
        return non_probes[-1]  # last non-probe SELECT actually run

    # 2) last executed SELECT (even if it looks like a probe)
    if selects:
        return selects[-1]

    # 3) No executed SELECT → parse fenced SQL and run the final one
    blocks = re.findall(r"```sql\s*(.*?)\s*```", output_text, flags=re.S | re.I)

    def _last_fenced_candidate():
        # prefer last non-probe fenced SELECT; else last fenced SELECT
        cands_np = [b.strip() for b in blocks if is_select(b) and not looks_like_probe(b)]
        if cands_np:
            return cands_np[-1]
        cands_any = [b.strip() for b in blocks if is_select(b)]
        return cands_any[-1] if cands_any else None

    sql = _last_fenced_candidate()
    if not sql:
        return None, None

    try:
        obs = run_query_callable(sql)
        return sql, obs
    except Exception as e:
        # If executing the fenced query fails, we still return the SQL string for debugging.
        return sql, f"Error executing fenced SQL: {e}"
    

class InferenceAgent:
    def __init__(self):
        # Initialize configuration, toolkit, and tools
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.sql_db, llm=self.config.llm_inference)
        self.tools = self.toolkit.get_tools()
        self.chat_prompt = self.create_chat_prompt()

        # Precompute lemma bags for each allowed table (both en/pt): { "en": {table: [lemmas]}, "pt": {table: [lemmas]} }
        self.table_bag_of_words: Dict[str, Optional[Dict[str, List[str]]]] = {"en": None, "pt": None}

        # Create an OpenAI-based agent with tools and prompt
        self.agent = create_openai_functions_agent(
            llm=self.config.llm_inference,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        # Configure the agent executor with runtime settings
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15,
            return_intermediate_steps=True
        )

    def run_query(self, q: str) -> str:
        # Execute a SQL query and handle errors if they occur
        try:
            return self.config.sql_db.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def labels(self, lang: str) -> Dict[str, str]:
        """Localized headings & instruction for EN/PT output."""
        if lang == "pt":
            return {
                "query": "Query executada",
                "results": "Resultados",
                "summary": "Resumo",
                "respond": "Responda em português de Portugal."
            }
        return {
            "query": "Query executed",
            "results": "Results",
            "summary": "Summary",
            "respond": "Respond in English."
        }

    def create_chat_prompt(self) -> ChatPromptTemplate:
        # Create a system prompt to guide the LLM's behavior and response format
        # accepts localized labels + tiny schema cards
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database inference expert for a SQL database.
            Your job is to answer questions by querying the database and providing clear, accurate results.

            Rules:

            TOOL BUDGET FOR EVENT TITLES:
            - At most TWO SQL tool calls in this turn:
            (1) one exact-title probe, then if 0 rows
            (2) one suggestions query.
            - If the exact probe returns 0 rows or an empty observation, DO NOT repeat it. Immediately run the suggestions query.
            - Never execute the same SQL twice.
            - If suggestions are returned, PRINT a bullet list and STOP (do not compute transactions in the same turn).
            - When printing a suggestions list, SKIP the "{label_query} / {label_results} / {label_summary}" block entirely.

            When the user question involves data from more than one table, you MUST perform JOIN operations, following the join rules.
            1. ONLY execute queries that retrieve data (NO INSERT/UPDATE/DELETE/DDL/CREATE)
            2. DO NOT provide analysis or recommendations - just the data
            3. Location resolution rule (CRITICAL): when the user filters by municipality or freguesia, you MUST join through the geo dimension (silver_dim_cim_concelho_freguesia):
                - First resolve the text to one or more DICOFRE codes in silver_dim_cim_concelho_freguesia (match on Municipio and Freguesia, case/accent-insensitive).
                - **CRITICAL DISTINCTION**: 
                    - For MUNICIPALITY/CONCELHO/MUNICIPIO: match on Municipio column only
                    - For PARISH/FRGUESIA: match on Freguesia column only
                - Then aggregate in silver_fact_transacoes_sibs filtering Fk_localizacao_DICOFRE IN the resolved DICOFRE set.
                - Use the join hint: silver_fact_transacoes_sibs.Fk_localizacao_DICOFRE = silver_dim_cim_concelho_freguesia.DICOFRE.
                - When filtering by a municipality, remember that a municipality contains multiple freguesias. You must sum transactions for all DICOFRE codes whose Municipio matches the requested name (case/accent-insensitive). Do not stop after the first match.
                - When filtering by a freguesia, use only DICOFRE codes whose Freguesia matches exactly.
                - Note that Municipio = Concelho = Municipality
            4. Only use joins that are explicitly documented in {join_hints}.
            5. Use only schema card fields "table" and "columns[*].name" as SQL identifiers; "title", "label", "table_aliases" and "description" are descriptive only
            6. Prefer the candidate tables. You may join additional tables from this allowed list, only if a join appears in join_hints: {allowed_tables}. If you use a non-candidate, cite the exact join hint line.
            7. Text comparisons (hard rule): For *any* comparison between a text column and a string literal,
                ALWAYS normalize both sides to case- and accent-insensitive form:
                Equality:
                    UPPER(<column>) COLLATE Latin1_General_CI_AI = UPPER(N'<value>')
                IN (...) list:
                    UPPER(<column>) COLLATE Latin1_General_CI_AI IN (UPPER(N'<v1>'), UPPER(N'<v2>'), ...)
                LIKE:
                    UPPER(<column>) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%<value>%')
                Joins on text:
                    ON UPPER(left_col)  COLLATE Latin1_General_CI_AI = UPPER(right_col)
                Never write plain:  <column> = '<value>'   (this is forbidden)                
            8. Unicode strings: when a literal may contain accents, ALWAYS prefix with N'...'.
                ## Example — normalization pattern
                BAD:
                    WHERE fk_concelho = 'Alcobaça'
                GOOD:
                    WHERE UPPER(fk_concelho) COLLATE Latin1_General_CI_AI = UPPER(N'Alcobaça')
            9. Aggregates must be null-safe: wrap counts/sums you return to users with COALESCE, e.g. SELECT COALESCE([SUM([Nº de Operações]), 0) AS total_transactions
            10. DATABASE LANGUAGE HANDLING (CRITICAL):
                The database contains PORTUGUESE text values. All categorical text (sectors, categories, types) is stored in Portuguese.

                WHEN USER ASKS IN ENGLISH:
                - You MUST translate English terms to Portuguese for database filters
                - Example: "services" → use "Serviços" in your WHERE clauses
                - Example: "commerce" → use "Comércio" in your WHERE clauses
                - Example: "industry" → use "Indústria" in your WHERE clauses

                WHEN USER ASKS IN PORTUGUESE:
                - Use the Portuguese terms directly

                ALWAYS probe first to verify the value exists in the database.
                If your first attempt returns 0 rows, try alternative translations or discover available values.

                REMEMBER: The database only understands Portuguese for text fields.
            11a. For any categorical filter, first run a probe SELECT (TOP 1) to confirm the value exists (case and accent-insensitive).
                If zero rows are found, do not proceed. Report that this dataset does not contain the necessary data to answer the user question.
            11b. Exception for EVENT TITLE queries:
                Do NOT “report no data” after a failed exact-title probe.
                You MUST run the suggestions query (Rule 21) and present a list if any titles exist.
            11c. Location probe (município/freguesia):
                - PROBE:
                    SELECT DISTINCT DICOFRE
                    FROM silver_dim_cim_concelho_freguesia
                    WHERE UPPER(Municipio) COLLATE Latin1_General_CI_AI = UPPER(N'<text>')
                    OR UPPER(Freguesia) COLLATE Latin1_General_CI_AI = UPPER(N'<text>');

                - Note that município and concelho are the same
                - ALWAYS collect **all matching DICOFRE codes** (not just the first one). 
                If multiple rows match (for example, several freguesias belong to one município), you MUST include **all their DICOFREs** in the final aggregation.

                - FINAL QUERY:
                    SELECT COALESCE(SUM(F.[Nº de Operações]), 0) AS total_transactions
                    FROM silver_fact_transacoes_sibs AS F
                    JOIN silver_dim_cim_concelho_freguesia AS G
                        ON F.Fk_localizacao_DICOFRE = G.DICOFRE
                    WHERE G.DICOFRE IN (<all matched DICOFREs>)
                    OR UPPER(G.Municipio) COLLATE Latin1_General_CI_AI = UPPER(N'<text>')
                    OR UPPER(G.Freguesia) COLLATE Latin1_General_CI_AI = UPPER(N'<text>');
            12. After probing, you MUST execute the FINAL aggregate query via the SQL tool. Show only the FINAL SQL in {label_query} (do not show probe SQL there).
            13. Do NOT print “{label_query} / {label_results}” unless you actually executed that SQL with the tool.
                All numbers in {label_results} must come from a tool observation, not from reasoning.
            14. For requests about the number of transactions (pt: número de transações, quantas transações), always compute SUM([Nº de Operações]) from silver_fact_transacoes_sibs.
            15. DO NOT assume the measure is "transactions". Only compute SUM([Nº de Operações]) when the user explicitly mentions transactions/payments/movements/operations (pt: transações/pagamentos/movimentos/operações). If the question asks for the number of entities (e.g., establishments like “alojamentos”, companies, venues) and there is no table with those entities, STOP and respond that the dataset does not contain the necessary data to answer that question. Do not reinterpret the question as transactions.
            16. The silver_fact_transacoes_sibs table contains data aggregated by week days and weekends. The 'Fk_date' column corresponds to the monday of that week/weekend. The column 'Tipo de dia' indicates if the row corresponds to transactions made during the week starting on that monday (if "Dia de Semana") or if it corresponds to the transactions made during the weekend of that week (if "Fim de Semana")  
                When the user question involves transaction dates, first check if it is regarding week days or weekends. 
                - if it is regarding week days, use the date from the monday of that corresponding week and filter 'Tipo de dia' by "Dia de Semana"
                - if it is regarding a weekend, use the date from the monday of that corresponding week and filter 'Tipo de dia' by "Fim de semana"
                - if it is regarding the entire week (week days + weekend), use both rows for that corresponding monday ("Dia de Semana" + "Fim de semana")
                - if the question is regarding an event, which has start and end dates:
                    - check if the duration of the event correspond to a week(s), weekday(s), or weekend; you can use the silver_dim_data table to find out what types of days they are (through column 'fim_de_semana'), or you can use the silver_fact_transacoes_sibs table, by going through column 'Fk_Date", that corresponds only to mondays, and column 'Tipo de dia', which specifies if the data from that row corresponds to the week days of that week or to the weekend.
                    - from the silver_events_oeste table, you can use the columns 'startDate', 'endDate' or 'monday_start_date' and 'monday_end_date' (if joining with the transactions table)
            17. When reporting results for an event related to transactions, always include a sentence in the summary stating whether the values refer to the week(s) or the weekend(s) during which the event took place. Use these rules:
                - If Tipo de dia = 'Dia de Semana' and duration (endDate - startDate) <= 5 days → say: 'Note that these results refer to the week in which the event occurred.'
                - If Tipo de dia = 'Dia de Semana' and duration > 5 days → say: 'Note that these results refer to the weeks in which the event occurred.'
                - If Tipo de dia = 'Fim de semana' → say: 'Note that these results refer to the weekend in which the event occurred.'
                - Translate the sentence into the user's language (PT/EN) as required.
            17. When asked about transaction sectors, FIRST probe silver_dim_transacoes_setor to find the sector specified by the user (case- and accent-insensitive) across columns CSA Nivel 1, CSA Nivel 2, CSA Nivel 3. If there are more than 1 level with the same name, show the user the options he has and ask him to choose which level he wants.
            18. For event questions that require time information, if the question can be answered using only the silver_events_oeste table, use the columns 'startDate' and 'endDate'
            19. For event questions that need transactions data, first probe the event row by title (case and accent insensitive) to get: Municipio, monday_start_date, monday_end_date, [Tipo de dia]
            20. When answering questions about the value of transactions, YOU MUST USE the unit €. Do not use "unidades montárias" or something similar.
            21. When answering questions about specific events, include in your response the startDate and endDate of the event if they are different. If startDate and endDate are the same, it means that the event occurred in just 1 day, so include that day in your response.
            22. When joining silver_fact_transacoes_sibs with silver_events_oeste, you should use the columns Municipio, monday_start_date, monday_end_date, and [Tipo de dia]. The join between these tables through the Municipio column MUST pass through the silver_dim_cim_concelho_freguesia table to get the DICOFRE codes.
                do not forget to join on silver_fact_transacoes_sibs.[Tipo de dia] = silver_events_oeste.[Tipo de dia]
                
                if you want you can create an auxiliary table from silver_events_oeste, for example like this:
                WITH event AS (
                    SELECT DISTINCT title, Municipio, monday_start_date, monday_end_date, [Tipo de dia]
                    FROM dbo.silver_events_oeste
                )
                FROM dbo.silver_fact_transacoes_sibs AS F
                JOIN event AS E
                ON F.Fk_Date BETWEEN E.monday_start_date AND E.monday_end_date
                JOIN dbo.silver_dim_cim_concelho_freguesia AS G
                ON F.Fk_localizacao_DICOFRE = G.DICOFRE
                WHERE UPPER(G.Municipio) COLLATE Latin1_General_CI_AI
                    = UPPER(E.Municipio) COLLATE Latin1_General_CI_AI
                    AND F.[Tipo de dia] = E.[Tipo de dia]
            23. Event title resolution (single turn, no loops)
                a) Exact probe (run ONCE only):
                SELECT TOP 1 title, Municipio, monday_start_date, monday_end_date, [Tipo de dia]
                FROM silver_events_oeste
                WHERE UPPER(title) COLLATE Latin1_General_CI_AI = UPPER(N'{user_title_original}');

                Escaping: use Unicode N'…' and escape single quotes (' → '').
                If the observation is empty or indicates zero rows, treat it as 0 rows and DO NOT retry.

                b) Suggestions query (run ONCE if exact probe returns 0 rows):
                    - Tokenize the user’s title into words/numbers (keep digits; strip quotes/spaces).
                    - Match with OR across tokens using case- and accent-insensitive LIKE.
                    - Rank by:
                        (1) number of matched tokens (more first),
                        (2) LEN(title) (shorter first),
                        (3) title (alphabetical).
                    - Return up to TOP 10 DISTINCT titles.

                    Template (adapt tokens dynamically):
                    SELECT TOP 10 title
                    FROM (
                        SELECT DISTINCT
                                title,
                                (/* one CASE WHEN per token */
                                CASE WHEN UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%<tok1>%') THEN 1 ELSE 0 END
                            + CASE WHEN UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%<tok2>%') THEN 1 ELSE 0 END
                                /* … repeat per token … */
                                ) AS match_score,
                                LEN(title) AS len_title
                        FROM silver_events_oeste
                        WHERE UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%<tok1>%')
                            OR UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%<tok2>%')
                            /* … repeat per token … */
                    ) q
                    ORDER BY match_score DESC, len_title, title;

                    Replace tok1, tok2, etc. with words from the user's event title ({user_title_original}).

                    Notes:
                    • Do NOT use CONTAINS or full-text search.
                    • If user tokens contain % or _, treat them as wildcards (do not double-escape).
                    • DO NOT MAKE UP SUGGESTIONS.

                c) If suggestions exist:
                Output the suggestions as a numbered or bulleted list IN YOUR RESPONSE and STOP (no final aggregate in this turn).
                Use the user’s language. Exact format (no code fences):

                I couldn’t find an exact match for "<user text>".
                Did you mean one of these? Reply with the exact event title:
                - <title 1>
                - <title 2>
                - <title 3>
                …

                (PT)
                Não encontrei uma correspondência exata para "<texto do utilizador>".
                Queria dizer um destes? Responda com o título exato:
                - <título 1>
                - <título 2>
                - <título 3>
                …

                - When showing event title suggestions, you MUST include the actual titles in your response.
                BAD:
                I couldn't find an exact match. Could you confirm the exact event title or choose from the suggestions provided?
                GOOD:
                I couldn't find an exact match for "Biblioteca Fora de Portas." Did you mean one of these?

                *results from suggestions SQL query*
                ...

                d) If no suggestions exist:
                Only then state that the dataset does not contain the necessary data to answer the question.

                CRITICAL: NEVER make up event titles
                When the exact match query returns 0 rows, you MUST:

                1. Run the suggestions query to get REAL titles from the database
                2. Show ONLY the titles returned by the database
                3. NEVER invent or hallucinate event titles

                Example workflow:
                - Exact match: SELECT ... WHERE title = 'User Title' → returns 0 rows
                - Suggestions: SELECT ... WHERE title LIKE '%token1%' OR ... → returns real database titles
                - Show ONLY the real database titles in your response

                CRITICAL FIX: Event title clarification workflow
                When the user provides an exact event title from suggestions, you MUST:

                1. Run ONLY the EXACT match query (never the suggestions query):
                    SELECT TOP 1 title, Municipio, monday_start_date, monday_end_date, [Tipo de dia]
                    FROM silver_events_oeste
                    WHERE UPPER(title) COLLATE Latin1_General_CI_AI = UPPER(N'<the exact title provided by the user>')

                2. If the exact match returns the event details, THEN proceed to compute transactions (if the question is about transactions)

                3. NEVER run the suggestions query when the user has already provided an exact title

                Example for clarification response:
                User provides: "Biblioteca Fora de Portas" em Abrigada com atividades para toda a família!
                You MUST run: SELECT ... WHERE UPPER(title) = UPPER(N'"Biblioteca Fora de Portas" em Abrigada com atividades para toda a família!') ...
            24. Do not write “Query executed / Results” unless you actually called the SQL tool.
            25. Do NOT validate municipalities/freguesias against silver_events_oeste. That table is not the authority for locations; use silver_dim_cim_concelho_freguesia.
            26. After probing, you MUST execute the FINAL aggregate query with the SQL tool. Do not write “Query executed” for SQL that you did not run via the tool.
            27. Format responses as:
               {label_query}: [the SQL query used]
               {label_results}: [the query results]
               {label_summary}: [brief factual summary of the findings]
            28. {respond_language_instruction}

            Schema cards (concise, read-only context for relevant tables):
            {schema_cards}

            Allowed joins (column = column):
            {join_hints}

            ### Few-shot: inexact event title
            User: How many transactions were made at the event Sonic 3?
            Assistant (tool): -- exact probe
            SELECT TOP 1 title, Municipio, monday_start_date, monday_end_date, [Tipo de dia]
            FROM silver_events_oeste
            WHERE UPPER(title) COLLATE Latin1_General_CI_AI = UPPER(N'Sonic 3');

            -- Observation: (empty)

            Assistant (tool): -- suggestions (tokens: SONIC, 3)
            SELECT TOP 10 title
            FROM (
            SELECT DISTINCT
                    title,
                    (CASE WHEN UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%SONIC%') THEN 1 ELSE 0 END
                    +CASE WHEN UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%3%')     THEN 1 ELSE 0 END) AS match_score,
                    LEN(title) AS len_title
            FROM silver_events_oeste
            WHERE UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%SONIC%')
                OR UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%3%')
            ) q
            ORDER BY match_score DESC, len_title, title;

            -- Observation: 
            -- | titles |

            Assistant (final to user, no {label_query}/{label_results}):
            I couldn’t find an exact match for "Sonic 3".
            Did you mean one of these? Reply with the exact event title:
            - titles found
            
            -- Probe (Óbidos)
            SELECT TOP 1 DICOFRE
            FROM silver_dim_cim_concelho_freguesia
            WHERE UPPER(Municipio) COLLATE Latin1_General_CI_AI = UPPER(N'Óbidos')
            OR UPPER(Freguesia) COLLATE Latin1_General_CI_AI = UPPER(N'Óbidos');

            -- Final aggregate (joining through the geo dim)
            SELECT COALESCE(SUM(F.[Nº de Operações]), 0) AS total_transactions
            FROM silver_fact_transacoes_sibs AS F
            JOIN silver_dim_cim_concelho_freguesia AS G
            ON F.Fk_localizacao_DICOFRE = G.DICOFRE
            WHERE UPPER(G.Municipio)  COLLATE Latin1_General_CI_AI = UPPER(N'Óbidos')
            OR UPPER(G.Freguesia) COLLATE Latin1_General_CI_AI = UPPER(N'Óbidos');

            -- Number of transactions during event (Sonic 3: O filme (VP))
            WITH event AS (
                SELECT DISTINCT Municipio, monday_start_date, monday_end_date, [Tipo de dia]
                FROM dbo.silver_events_oeste
                WHERE UPPER(title) COLLATE Latin1_General_CI_AI = UPPER(N'Sonic 3: O filme (VP)')
            )
            SELECT
                COALESCE(SUM(F.[Nº de Operações]), 0) AS total_transactions
            FROM dbo.silver_fact_transacoes_sibs AS F
            JOIN event AS E
            ON F.Fk_Date BETWEEN E.monday_start_date AND E.monday_end_date
            JOIN dbo.silver_dim_cim_concelho_freguesia AS G
            ON F.Fk_localizacao_DICOFRE = G.DICOFRE
            WHERE UPPER(G.Municipio) COLLATE Latin1_General_CI_AI
                = UPPER(E.Municipio) COLLATE Latin1_General_CI_AI
                AND F.[Tipo de dia] = E.[Tipo de dia];
            """
        )

        # Create a template for user-provided input
        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([system_message, 
                                                 human_message,
                                                 MessagesPlaceholder(variable_name="agent_scratchpad")])

    def table_text(self, table_key: str, lang: str) -> str:
        """Compose a language-aware textual summary for one table using only column labels, and synonyms.
        """
        tab_meta = self.config.table_meta.get(table_key, {})
        col_catalog = self.config.column_catalog.get(table_key, {})

        parts: list[str] = []
        # table title + synonyms + content (in detected language)
        parts.append(tab_meta.get(f"title_{lang}"))
        parts.extend(tab_meta.get(f"synonyms_{lang}"))
        parts.append(tab_meta.get(f"content_{lang}"))

        # columns: labels, synonyms
        for _, labels in col_catalog.items():
            parts.append(labels.get(lang, ""))
            parts.extend(labels.get(f"synonyms_{lang}", []))        # synonyms in same lang
        return " ".join(p for p in parts if p)

    # Precompute lemma bags for all allowed tables (pt/en) using clean_text
    def build_table_bag_of_words(self, lang: str) -> dict[str, list[str]]:
        """Return {table_name: [lemmas...]} for ALL allowed tables, in `lang`."""
        bags: dict[str, list[str]] = {}
        for t in self.config.table_names:
            txt = self.table_text(t, lang)
            bags[t] = clean_text(txt, lang=lang, nlp_pt=nlp_pt)
        return bags
    
    def rank_tables(self, question: str, lang: str, top_k: int = 4, mode: str = "recall", min_score: float = 0.08, 
                    allow_language_fallback: bool = True) -> list[str]:
        """Overlap between question lemmas and EACH table's bag → top-K tables."""
        q = set(clean_text(question, lang=lang, nlp_pt=nlp_pt))
        if not q:
            # hand control to the LLM to choose the tables itself
            return []

        # get main language bag of words
        bags = self.build_table_bag_of_words(lang)
        if not bags:
            return []

        scores: list[tuple[str, float]] = []    # a list that will hold pairs (table_name (str), similarity_score(float))
        for t, bag in bags.items():          # iterate per-table bag for the detected language. t: table name; bag: list of lemmas for that table
            b = set(bag)    # convert the list to a set -> deduplicate tokens
            inter = len(q & b)  # size of the intersection between the question lemmas and the table lemmas (how many lemmas they share)
            if mode == "jaccard": 
                denom = (len(q | b) or 1)   # Jaccard similarity: |∩| / |∪|; or 1 avoids division by 0 if a set is empty
            else: # recall-like score |∩| / |Q|; or 1 avoids division by 0 if a set is empty
                denom = (len(q) or 1)
            scores.append((t, inter / denom))   # compute the similarity and store it together with the table name
        scores.sort(key=lambda x: x[1], reverse=True)   # sort tables descending by their similarity score -> best matches come first

        # Optional fallback to the other language if matches are very weak
        if allow_language_fallback and (not scores or scores[0][1] < min_score):
            other = "pt" if lang == "en" else "en"
            bags_other = self.build_table_bag_of_words(other)
            q_other = set(clean_text(question, lang=other, nlp_pt=nlp_pt))
            alt = []
            for t, bag in bags_other.items():
                b = set(bag)
                inter = len(q_other & b)
                if mode == "jaccard": 
                    denom = (len(q_other | b) or 1)
                else:
                    denom = (len(q_other) or 1)
                alt.append((t, inter / denom))
            alt.sort(key=lambda x: x[1], reverse=True)

            if alt and (not scores or alt[0][1] >= scores[0][1]):
                scores = alt
        return [t for t, _ in scores[:top_k]]

    def make_schema_cards(self, candidates: List[str], lang: str) -> str:
        cards = []
        for t in candidates:
            tmeta = self.config.table_meta.get(t, {})
            cols = self.config.column_catalog.get(t, {})

            col_items = [{"name": name, "label": lab.get(lang, "")}
                    for name, lab in cols.items()]
            
            cards.append({
                "table": t,
                "title": tmeta.get(f"title_{lang}", ""),
                "table_aliases": tmeta.get(f"synonyms_{lang}", []),
                "description": (tmeta.get(f"content_{lang}", "") or "").splitlines()[0][:200],
                "columns": col_items
            })
        return json.dumps({"schema_cards": cards}, ensure_ascii=False)

    def query(self, text: str, db_graph, candidates_override: Optional[List[str]] = None) -> str:
        """
        Detect language → rank tables offline → pass only top-K schema cards to the LLM → respond in the same language.
        """
        try:
            # 1) detect language & labels
            lang = detect_lang(text)  # 'en' or 'pt'
            lbl = self.labels(lang)
            top_k = 4
            user_title = None
            user_title_original = None

            if looks_like_event_q(text):
                # try to grab what the user typed after the word event triggers words
                # normalize triggers
                # unidecode -> remove acentos 
                # lower -> tudo minusculas
                # strip -> tira espaços no fim e início. Cuidado especial caso trigger word tenha duas palavras
                norm_triggers = [unidecode(t).lower().strip() for t in event_triggers if t]
                # previne existência de pontuação nos triggers
                escaped = [re.escape(t) for t in norm_triggers]

                # normalize the user text once
                norm_text = unidecode(text).lower()

                # Try to find CSV-style doubled-quote content first (keeps internal quotes when doubled)
                # 1) double-quoted with doubled internal quotes: " ( (?: [^"] | "" )+ ) "
                dq_pattern = r'"((?:[^"]|"")+)"'
                # 2) single-quoted with doubled internal single-quotes: '((?:[^\']|\'\')+)' 
                sq_pattern = r"'((?:[^']|'' )+)'".replace("'' )", "'' )")  # keep literal but avoid formatting issues
                # 3) smart quotes (no escaping inside)
                smart_pattern = r'[“”\u201c\u201d]([^“”\n]+)[“”\u201c\u201d]'

                # Try double-quotes then single-quotes then smart quotes on the ORIGINAL text (preserve accents)
                m_quoted = re.search(dq_pattern, text) or re.search(sq_pattern, text) or re.search(smart_pattern, text)

                if m_quoted:
                    # Extract the captured content exactly as typed (preserve inner doubled quotes)
                    raw = m_quoted.group(1).strip(" \t\n'\"")
                    # Convert doubled quotes to single literal quote inside title (CSV-style)
                    # If double-quote pattern matched, replace "" -> "
                    raw = raw.replace('""', '"').replace("''", "'")
                    user_title_original = raw
                    # normalized version for matching/SQL-building
                    user_title = unidecode(user_title_original).lower().strip()
                else:
                    # No explicit quoted title found.
                    # Do NOT assume the rest of the sentence is a title.
                    # Clear user_title so event-title clarification flow won't trigger.
                    user_title = None
                    user_title_original = None

                # NOTE: the rest of your logic (exact probe / suggestions) should run only when user_title is not None.
                # If you want a fallback behaviour (e.g., heuristic or ask for clarification) implement that separately.

                # # pattern = r"\b(?:" + "|".join(escaped) + r")\b(?:[:\s\-–—]+)?([^\n]+)"
                # m = re.search(pattern, norm_text)
                # if m:
                #     user_title = m.group(1).strip(' \t\n\'"')
                #     # Note: user_title is normalized (no accents). If you need the original spelling,
                #     # capture from the original `text` using the match span:
                #     start, end = m.span(1)
                #     user_title_original = text[start:end].strip(" \t\n'\"?")

                # m = re.search(r"\bevent\s+(.+)", text, flags=re.I)
                # user_title = (m.group(1) if m else text).strip()
                # user_title = re.sub(r'^[\'"]|[\'"]$', '', user_title)  # strip edge quotes
                # punctuation cleaning
                # user_title = self.clean_title_for_matching(user_title)
                    
            # 2) pick top-K candidates offline
            if candidates_override is not None:
                candidates = list(candidates_override) if isinstance(candidates_override, list) else []
            else:
                candidates = self.rank_tables(text, lang, top_k)

            # candidates = self.rank_tables(text, lang, top_k)

            if looks_like_event_q(text) and event_table in selected_tables:
                candidates = boost_events_first(candidates, top_k)
            
            # 3) allowed tables
            allowed_tables_str = ", ".join(selected_tables)

            # 4) join hints
            join_hints = make_join_hints(tables=candidates or selected_tables)

            if candidates:
                # 5) tiny schema cards
                schema_cards_json = self.make_schema_cards(candidates, lang)

                # 6) enhanced user input
                enhanced_prompt = f"""
                    Database Structure Analysis:
                    - Candidate Tables: {candidates}

                    User Question: {text}

                    Use this structural information to form an accurate query.
                    """
            else:   # hand-off path: no schema cards; let LLM choose from allowed list
                schema_cards_json = json.dumps({"schema_cards": []}, ensure_ascii=False)
                enhanced_prompt = f"""
                    No schema cards were computed for this question (query terms were too generic after cleaning).
                    Identify the correct tables yourself from this allowed list: {allowed_tables_str}

                    User Question: {text}

                    Write a safe, read-only SQL query. If you join, use only documented joins from the hints.
                    """
                
            logger.info("TITULO ORIGINAL")
            logger.info(user_title_original)

            # 7) invoke the agent with all template variables
            result = self.agent_executor.invoke({
                "input": enhanced_prompt,
                "label_query": lbl["query"],
                "label_results": lbl["results"],
                "label_summary": lbl["summary"],
                "respond_language_instruction": lbl["respond"],
                "schema_cards": schema_cards_json,
                "join_hints": join_hints,
                "allowed_tables": allowed_tables_str,
                "user_title_original": user_title_original
            })
            
            output_text = result["output"]
            logger.info("OUTPUT TEXT")
            logger.info(output_text)
            steps = result.get("intermediate_steps", [])
            # dump_steps(steps)

            last_sql = None
            last_answer = None

            # Accept multiple tool names and dict inputs
            sql_tools = {"sql_db_query", "query_sql_db", "sql_db_run"}

            executed_sql: list[tuple[str, Any]] = []

            for action, observation in steps:
                logger.info("ACTION AND OBSERVATION")
                logger.info(action)
                logger.info(observation)
                tool_name = getattr(action, "tool", "")
                tool_input = getattr(action, "tool_input", None)

                sql_candidate = None
                if isinstance(tool_input, str):
                    sql_candidate = tool_input
                elif isinstance(tool_input, dict):
                    # common keys used by LangChain tools/ my wrappers
                    sql_candidate = (
                        tool_input.get("query")
                        or tool_input.get("sql")
                        or tool_input.get("statement")
                        or tool_input.get("input")
                    )

                if tool_name in sql_tools and sql_candidate:
                    executed_sql.append((sql_candidate, observation))
                    
            # If an exact-title probe ran against silver_events_oeste and returned no rows,
            # automatically run the suggestions query once (per policy) so suggestions come
            # from the database instead of being hallucinated by the LLM.
            try:
                if user_title_original:
                    # Find a matching exact-title probe in executed_sql
                    probe_sql = None
                    probe_obs = None
                    for sql, obs in executed_sql:
                        s = (sql or "").lower()
                        if "from silver_events_oeste" in s and "upper(title)" in s and "= upper(n'" in s:
                            probe_sql = sql
                            probe_obs = obs
                    # If we found a probe and it returned no rows, and no suggestions have been run yet,
                    # build and run the suggestions SQL once.
                    if probe_sql and (not probe_obs or (isinstance(probe_obs, str) and not probe_obs.strip())):
                        # Ensure we haven't already executed a suggestions query
                        already = any(("select top 10 title" in (q or "").lower() and "match_score" in (q or "").lower()) for q, _ in executed_sql)
                        if not already:
                            # Tokenize the original title into words/numbers; keep tokens as typed
                            toks = [t for t in re.findall(r"[\w%\d\u00C0-\u017F]+", user_title_original) if t.strip()]
                            if toks:
                                # escape single quotes in tokens (but keep %/_ wildcards if present)
                                def esc(tok):
                                    return tok.replace("'", "''")

                                case_clauses = []
                                where_clauses = []
                                for tok in toks:
                                    t = esc(tok)
                                    case_clauses.append(f"CASE WHEN UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%{t}%') THEN 1 ELSE 0 END")
                                    where_clauses.append(f"UPPER(title) COLLATE Latin1_General_CI_AI LIKE UPPER(N'%{t}%')")

                                case_expr = " + ".join(case_clauses)
                                where_expr = "\n            OR ".join(where_clauses)

                                suggestions_sql = (
                                    "SELECT TOP 10 title\n"
                                    "FROM (\n"
                                    "  SELECT DISTINCT\n"
                                    "    title,\n"
                                    f"    ({case_expr}) AS match_score,\n"
                                    "    LEN(title) AS len_title\n"
                                    "  FROM silver_events_oeste\n"
                                    "  WHERE " + where_expr + "\n"
                                    ") q\n"
                                    "ORDER BY match_score DESC, len_title, title;"
                                )

                                try:
                                    obs = self.run_query(suggestions_sql)
                                except Exception as e:
                                    obs = f"Error executing suggestions SQL: {e}"

                                # record that we ran it so pick_final_query can see it
                                executed_sql.append((suggestions_sql, obs))
                                # also expose it as the last_sql/last_answer (so Supervisor and caller can use it)
                                # Note: we do not short-circuit here; pick_final_query will decide what to prefer.
            except Exception:
                # Don't let suggestion-generation failures block the rest of the flow
                pass

            # Decide which one is the *final* query (prefers non-probes; else runs fenced aggregate)
            last_sql, last_answer = pick_final_query(
                executed_sql = executed_sql,
                output_text = output_text,
                run_query_callable = self.run_query)    # lets helpers execute the fenced FINAL if needed

            # return both the formatted output + structured bits so Supervisor can store them
            return {"text": output_text, "last_sql": last_sql, "last_answer": last_answer}

        except Exception as e:
            print(f"\n❌ Error in inference query: {str(e)}")
            return f"Error processing query: {str(e)}"