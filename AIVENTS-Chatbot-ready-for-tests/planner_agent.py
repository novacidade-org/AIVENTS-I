import re
from unidecode import unidecode
from sqlalchemy import text as sqltext
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from config import Config
from utils import get_chat_history, detect_lang, make_join_hints, get_long_term_memory
from logger import logger

# check for ambiguous locations in user question and not specification of municipality/parish
def build_ambiguous_geo_names() -> set[str]:
    """
    Return names that exist as BOTH freguesia and concelho.
    """

    DIM = "silver_dim_cim_concelho_freguesia"

    query = f"""
            WITH D AS (
            SELECT DISTINCT RTRIM(LTRIM(Freguesia)) AS n
            FROM {DIM}
            WHERE Freguesia IS NOT NULL
            ),
            C AS (
            SELECT DISTINCT RTRIM(LTRIM(Municipio)) AS n
            FROM {DIM}
            WHERE Municipio IS NOT NULL
            )
            SELECT D.n
            FROM D
            INNER JOIN C ON C.n = D.n;
            """
    
    names = set()
    with Config().db_engine.connect() as conn:
        for (n,) in conn.execute(sqltext(query)).fetchall():
                names.add(n)  # keep canonical form as stored in DB

        return names

ambiguous_locations = build_ambiguous_geo_names()

def norm(s: str) -> str:
    # lower, strip accents, trim, collapse inner whitespace
    s = unidecode((s or "").lower().strip())
    return re.sub(r"\s+", " ", s)

# Build a normalized index → keep original for nice prompts
ambiguous_idx = { norm(n): n for n in ambiguous_locations }

def find_ambiguous_name_in_question(question: str) -> str | None:
    """
    Return the original ambiguous toponym if the question contains one *and*
    the user did not already specify municipality/parish.
    """
    qn = norm(question)

    # If user already clarified with explicit terms, bail out
    clarification_terms = {
        "pt": ["concelho", "município", "municipio", "freguesia"],
        "en": ["municipality", "parish"]
    }

    # Check if user already specified location type
    for lang_terms in clarification_terms.values():
        for term in lang_terms:
            if term in qn:
                return None

    # Look for whole-phrase occurrence (handles multi-word names)
    for norm_name, original in ambiguous_idx.items():
        # Whole-word-ish boundaries around the phrase
        pattern = rf"(?<!\w){re.escape(norm_name)}(?!\w)"
        if re.search(pattern, qn):
            return original

    return None

class PlannerAgent:
    def __init__(self):
        # Initialize configuration and planner prompt
        self.config = Config()
        self.allowed_tables = self.config.selected_tables
        self.planner_prompt = self.create_planner_prompt()

        self.planner_chain = self.planner_prompt | self.config.llm_planner

        self.planner_with_mem = RunnableWithMessageHistory(
            self.planner_chain,
            get_chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def create_planner_prompt(self):
        # Define the system template for planning instructions
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query] - Use this prefix for database queries
        2. General: [response] - Use this prefix for friendly responses

        RULES:
        - Use ONLY these tables: {allowed_tables}{candidates_hint}
        - NEVER modify data (no INSERT/UPDATE/DELETE/DDL).
        - Prefer joins and filters that can be expressed with the listed tables.
        - When resolving municipality/freguesia filters, plan ONE join from the table you need with silver_dim_cim_concelho_freguesia using the documented join hint.
        - For location questions:
            - If user asks about "município", "concelho", or "municipality": filter ONLY on Municipio column
            - If user asks about "freguesia" or "parish": filter ONLY on Freguesia column  
            - NEVER mix filters - this causes incorrect results
            - Plan exactly ONE "Inference:" step that:
                1. joins silver_fact_transacoes_sibs to silver_dim_cim_concelho_freguesia on Fk_localizacao_DICOFRE = DICOFRE,
                2. filters on the CORRECT column based on user intent,
                3. computes SUM([Nº de Operações]).
        - Do NOT validate location existence in silver_events_oeste. Use the geo dimension only (silver_dim_cim_concelho_freguesia).
        - When asked about events and transactions, you should use silver_fact_transacoes_sibs, silver_events, and silver_dim_cim_concelho_freguesia tables, performing the needed joins to get the necessary data.
        - All string filters must be case- and accent-insensitive.
        - If you have a fact table (name contains "fact"), assume it contains the required measures; use it first.
        - Dimension tables (name contains "dim") are ONLY for:
            (a) filtering on attributes not present in the fact table, or
            (b) selecting human-readable labels not present in the fact table.
        - **Plan must contain at most ONE Inference step.** If any value must be validated (e.g., municipality text),
        the InferenceAgent will internally probe first and then run the final aggregate — do NOT emit a separate probe step.
        - Only use joins that are explicitly documented here (column = column): {join_hints}
        - When joining on text columns (e.g., names), normalize both sides using case- and accent-insensitive comparison
        - If the user's language differs from the table language, translate categorical values before filtering; when uncertain, plan to try both translated and original forms with OR and normalized comparisons.
        - When the user asks for the number of transactions (e.g., “number of transactions”, “transactions count”, pt: “número de transações”, “quantas transações”), compute SUM([Nº de Operações]) from silver_fact_transacoes_sibs.
        - DO NOT assume the measure is "transactions". Only compute SUM([Nº de Operações]) when the user explicitly mentions transactions/payments/movements/operations (pt: transações/pagamentos/movimentos/operações). If the question asks for the number of entities (e.g., establishments like “alojamentos”, companies, venues) and there is no table with those entities, STOP and respond that the dataset does not contain the necessary data to answer that question. Do not reinterpret the question as transactions.
        - For event questions, if the question can be answered using only the silver_events_oeste table, use the columns 'startDate' and 'endDate' for questions that need time information
        - For event questions that need transactions data, first probe the event row by title (case and accent insensitive) to get: Municipio, monday_start_date, monday_end_date, [Tipo de dia]
        - For event questions with inexact title: 
            - Plan exactly ONE Inference step that (i) tries the exact title, and if 0 rows (ii) returns up to 10 title suggestions
                using the multi-token OR LIKE query (case/accent-insensitive) described in the InferenceAgent rules.
            - Then plan ONE General step asking the user to pick one exact title.
            - Do NOT plan a transactions aggregation step in the same turn when suggestions are shown.
            - Do NOT “report no data” until after the suggestions query returns 0 rows.
        - Enforce the tool budget for inexact event titles:
            - Plan exactly one Inference step (exact probe → suggestions) and then one General step asking the user to choose.
            - Do not plan a transactions aggregation step in the same turn when suggestions are shown.
            - Do not “report no data” until after the suggestions query returns 0 rows.
        - When joining silver_fact_transacoes_sibs with silver_events, you should use the columns Municipio, monday_start_date, monday_end_date, [Tipo de dia]
        - When the user question involves transaction dates, use the date from the monday of that corresponding week and inform the user that you don't have data concerning specific days, only concerning that week or weekend.
        - When answering questions about transactions that involves dates, you need to mention that the results are from that week/weekend
        - When answering questions about specific events, include in your response the startDate and endDate of the event if they are different. If startDate and endDate are the same, it means that the event occurred in just 1 day, so include that day in your response.
        - When answering questions about the value of transactions, YOU MUST USE the unit €. Do not use "unidades montárias" or something similar.
        - If filtering by sector without an explicit ‘level’ label, plan a JOIN between the fact and silver_dim_transacoes_setor on Sk_Setor=Fk_Setor and find the options the user might be refering to.
        - Match the user's language: {response_language}.
        - Create a SINGLE, SEQUENTIAL plan where:
            - Each step should be exactly ONE line
            - Each step must start with either 'Inference:' or 'General:'
            - Steps must be in logical order
            - DO NOT repeat steps
            - Keep the plan minimal and focused

        Guidance for Inference steps (when applicable):
        - Name the target table(s) explicitly (from the allowed list).
        - If you propose a join, briefly state WHY it’s necessary (missing filter/label in fact).
        - List the key columns to SELECT.
        - State WHERE filters, GROUP BY, and ORDER BY if relevant.
        - If the user intent is unclear, add one 'General:' step to ask a single, crisp clarification question.

        Example format:
        Inference: Compute SUM([Nº de Operações]) from silver_fact_transacoes_sibs filtered for the requested municipality (case/accent-insensitive). If the municipality does not exist, stop and report no data.
        General: Provide the result in a friendly way
        """

        # Define the human message template for user input
        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([
            # SystemMessagePromptTemplate.from_template(system_template),
            ("system", system_template),
            ("system", "Long-term memory: {long_term_memory}"),
            MessagesPlaceholder(variable_name="history"),
            # HumanMessagePromptTemplate.from_template(human_template)
            ("human", human_template),
            #MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def create_plan(self, question: str, *, session_id: str, candidates: list[str] | None = None) -> list:
    # Generate a step-by-step plan to answer the given question
        try:
            # Language hint for the planner (keeps terminology consistent downstream)
            lang = detect_lang(question)  # 'en' or 'pt'
            response_language = "Portuguese (Portugal)" if lang == "pt" else "English"
            
            # Check if this is a clarified question with location type
            # This handles the case where we're creating a plan AFTER clarification
            if "(freguesia)" in question or "(concelho)" in question or "(parish)" in question or "(municipality)" in question:
                # Extract the original question and location type
                original_q = question.replace(" (freguesia)", "").replace(" (concelho)", "").replace(" (parish)", "").replace(" (municipality)", "")
                location_type = "parish" if any(x in question for x in ["(freguesia)", "(parish)"]) else "municipality"
                
                # Create a plan that uses the specified location type
                if location_type == "parish":
                    if lang == "pt":
                        return [f"Inference: Calcular SUM([Nº de Operações]) de silver_fact_transacoes_sibs, fazendo um JOIN com silver_dim_cim_concelho_freguesia em Fk_localizacao_DICOFRE=DICOFRE e filtrado por Freguesia = '{original_q.split()[-1]}' (comparação case/accent-insensitive)"]
                    else:
                        return [f"Inference: Compute SUM([Nº de Operações]) from silver_fact_transacoes_sibs, fazendo um JOIN com silver_dim_cim_concelho_freguesia em Fk_localizacao_DICOFRE=DICOFRE e filtrado por Freguesia = '{original_q.split()[-1]}' (case/accent-insensitive comparison)"]
                else:
                    if lang == "pt":
                        return [f"Inference: Calcular SUM([Nº de Operações]) de silver_fact_transacoes_sibs, fazendo um JOIN com silver_dim_cim_concelho_freguesia em Fk_localizacao_DICOFRE=DICOFRE e filtrado por Municipio = '{original_q.split()[-1]}' (comparação case/accent-insensitive)"]
                    else:
                        return [f"Inference: Compute SUM([Nº de Operações]) from silver_fact_transacoes_sibs, fazendo um JOIN com silver_dim_cim_concelho_freguesia em Fk_localizacao_DICOFRE=DICOFRE e filtrado por Municipio = '{original_q.split()[-1]}' (case/accent-insensitive comparison)"]

            # Check for generic clarification format [clarification: ...]
            if "[clarification:" in question:
                # Extract the original question and clarification
                parts = question.split("[clarification:")
                original_q = parts[0].strip()
                clarification = parts[1].replace("]", "").strip()
                
                # The planner will naturally interpret the combined context
                # No special handling needed - let the LLM figure it out
                logger.info(f"Processing generic clarification: original='{original_q}', clarification='{clarification}'")

            # deal with ambiguous locations for initial questions
            amb = find_ambiguous_name_in_question(question)

            if amb:
                if lang == "pt":
                    return [f"General: Quer dizer a freguesia de {amb} ou o concelho de {amb}?"]
                else:
                    return [f"General: Do you mean the parish of {amb} or the municipality of {amb}?"]
            
            # Optional nudge: show candidate tables to the planner
            if candidates:
                candidates_hint = f"\n- Prioritize these relevant tables if appropriate: {', '.join(candidates)}"
            else:
                candidates_hint = ""

            join_hints = make_join_hints(tables=candidates or self.allowed_tables)

            long_term_mem = get_long_term_memory(session_id)

            msg = {
                "question": question,                               # human input for history
                "allowed_tables": ", ".join(self.allowed_tables),
                "candidates_hint": candidates_hint,
                "response_language": response_language,
                "join_hints": join_hints,
                "long_term_memory": long_term_mem,
            }

            logger.info(f"Creating plan for question: {question}")

            response = self.planner_with_mem.invoke(
                msg,
                config={"configurable": {"session_id": session_id}}
            )

            # Normalize the plan to a list of lines that start with Inference:/General:
            steps = [
                line.strip()
                for line in response.content.splitlines()
                if line.strip() and (line.strip().startswith("Inference:") or line.strip().startswith("General:"))
            ]

            # Provide a fallback message if no steps are returned
            if not steps:
                if lang == "pt":
                    return ["General: Pode especificar melhor a pergunta para eu selecionar as tabelas corretas?"]
                return ["General: Could you clarify your question so I can pick the right tables?"]
            
            # collapse to at most one Inference step
            inference_seen = False
            filtered = []
            for s in steps:
                if s.startswith("Inference:"):
                    if inference_seen:
                        logger.info("[planner] Dropping extra Inference step: %s", s)
                        continue
                    inference_seen = True
                filtered.append(s)
            steps = filtered

            return steps

        except Exception as e:
            # Log and handle errors during plan creation
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]