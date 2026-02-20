from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
import re
from config import Config
from inference_agent import InferenceAgent
from planner_agent import PlannerAgent
from discovery_agent import DiscoveryAgent
from conversation_state import ConversationState
from logger import logger
from constants import event_table, selected_tables
from utils import get_chat_history, detect_lang, looks_like_event_q, boost_events_first, get_long_term_memory, is_select, long_term_memory, looks_like_probe

def update_long_term_memory(session_id: str, input: str, output: str):
    if session_id not in long_term_memory:
        long_term_memory[session_id] = []
    if len(input) > 20:     # simple logic: store inputs longer than 20 characters
        long_term_memory[session_id].append(f"User said: {input}")
    if len(long_term_memory[session_id]) > 5:   # keep only last 5 memories
        long_term_memory[session_id] = long_term_memory[session_id][-5:]

class SupervisorAgent:
    def __init__(self):
        # Initialize configuration and agents
        self.config = Config()
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()
        self.discovery_agent = DiscoveryAgent()

        # Prompts for different types of responses
        self.db_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Steps & Logs: {db_results}
            Final SQL: {last_sql}
            Final Result: {last_answer}

            Rules:
            1. The Final SQL and Final Result are the single source of truth.
            2. If there is any conflict in earlier logs, ignore them and use Final Result.
            3. Output one sentence, with no bullets, code, or headings, except when specifically asked for, or when {last_answer} contains more than 2 values. In that case, you should use a concise bullet (or numbered) list.
            5. Respond in: {response_language}
            """),
            ("system", "Long-term memory: {long_term_memory}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        self.chat_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant.
            Respond naturally to the user's message.
            Keep responses brief and friendly.
            Don't make up information about weather, traffic, or other external data.
            Respond in: {response_language}
            """),
            ("system", "Long-term memory: {long_term_memory}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])

        self.db_chain = self.db_response_prompt | self.config.llm_supervisor
        self.db_with_mem = RunnableWithMessageHistory(
            self.db_chain,
            get_chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )

        self.chat_chain = self.chat_response_prompt | self.config.llm_supervisor
        self.chat_with_mem = RunnableWithMessageHistory(
            self.chat_chain,
            get_chat_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def create_plan(self, state: ConversationState) -> ConversationState:
        q = state['question']
        lang = detect_lang(q)
        top_k = 4

        # Use existing candidates if available (from clarification context)
        candidates = state.get("candidates")
        logger.info("CANDIDATE TABLES")
        logger.info(candidates)
        if not candidates:
            candidates = self.inference_agent.rank_tables(q, lang, top_k=top_k)
            logger.info("CANDIDATE TABLES WITH RANK TABLES")
            logger.info(candidates)
        if candidates == []:
            candidates = selected_tables
            logger.info("CANDIDATE TABLES - SELECTED TABLES")
            logger.info(candidates)

        if looks_like_event_q(q) and event_table in selected_tables:
            candidates = boost_events_first(candidates, top_k)

        # Generate a plan using the PlannerAgent
        plan = self.planner_agent.create_plan(
            question = q,
            session_id=state["user_id"],
            candidates = candidates,
            # state=state
        )

        # Log the plan, separating inference and general steps
        logger.info("Generated plan:")
        inference_steps = [step for step in plan if step.startswith('Inference:')]
        general_steps = [step for step in plan if step.startswith('General:')]

        if inference_steps:
            logger.info("Inference Steps:")
            for i, step in enumerate(inference_steps, 1):
                logger.info(f"  {i}. {step}")
        if general_steps:
            logger.info("General Steps:")
            for i, step in enumerate(general_steps, 1):
                logger.info(f"  {i}. {step}")

        return {
            **state,
            "plan": plan,
            "lang": lang,
            "candidates": candidates
        }

    def execute_plan(self, state: ConversationState) -> ConversationState:
        """
        Execute the generated plan. For Inference steps, call the InferenceAgent once.
        If the agent only executed a probe (e.g., SELECT TOP 1 ...) or did not execute the
        final aggregate via the tool, parse the last fenced ```sql``` aggregate from its text
        and execute it here as a supervisor fallback. Persist that as last_sql/last_answer.
        """
        
        # Execute the generated plan step by step
        results = []

        # carry previous states if any (useful for multi-step plans)
        last_sql = state.get("last_sql")
        last_answer = state.get("last_answer")
        
        try:
            plan = state.get("plan") or []
            saw_inference = False

            for step in plan:
                if not isinstance(step, str) or ':' not in step:
                    continue

                step_type, content = step.split(':', 1)
                step_type = step_type.strip().lower()
                content = content.strip()

                if step_type == 'inference':
                    saw_inference = True
                    # Handle inference steps using the InferenceAgent
                    try:
                        # 1) let the inference agent do its thing (may probe + print final SQL, or may run it)
                        result = self.inference_agent.query(state["question"], state.get('db_graph'), candidates_override=state.get("candidates"))
                        logger.info("RESULT DA QUERY")
                        logger.info(result)
                        
                        # Pretty text for db_results
                        pretty_text = result.get("text", "") if isinstance(result, dict) else (result or "")
                        results.append(f"Step: {step}\nResult: {pretty_text}")

                        # Structured outputs (what the agent thinks is final)
                        cand_sql = (result.get("last_sql") if isinstance(result, dict) else None)
                        cand_ans = (result.get("last_answer") if isinstance(result, dict) else None)

                        # Normalize answer to string
                        if cand_sql:
                            last_sql = cand_sql
                        if cand_ans is not None:
                            last_answer = cand_ans if isinstance(cand_ans, str) else str(cand_ans)

                        # 2) Supervisor fallback: if nothing executed OR only a probe, run the fenced final aggregate ourselves
                        need_fallback = (not cand_sql) or looks_like_probe(cand_sql)

                        if need_fallback:
                            # Pull all fenced SQL blocks from the agent's text
                            blocks = re.findall(r"```sql\s*(.*?)\s*```", pretty_text, flags=re.S | re.I)
                            # Keep only SELECTs that don't look like probes (i.e., aggregates or GROUP BY)
                            candidates = [b.strip() for b in blocks if is_select(b) and not looks_like_probe(b)]

                            if candidates:
                                # Heuristic: take the last non-probe block as the final aggregate
                                final_sql = candidates[-1]
                                try:
                                    obs = self.inference_agent.run_query(final_sql)
                                    # Persist the *true* final query + observation
                                    last_sql = final_sql
                                    last_answer = obs if isinstance(obs, str) else str(obs)

                                    # Append to the trace so it's visible in db_results
                                    results.append(
                                        "Step: Supervisor fallback\n"
                                        "Result: Executed final aggregate from fenced SQL.\n"
                                        f"SQL:\n```sql\n{final_sql}\n```\n"
                                        f"Observation:\n{obs}"
                                    )
                                except Exception as e:
                                    results.append(f"Step: Supervisor fallback\nError: Failed to run fenced SQL: {e}")

                    except Exception as e:
                        logger.error(f"Error in inference step: {str(e)}", exc_info=True)
                        results.append(f"Step: {step}\nError: Query failed - {str(e)}")
                else:
                    # Handle general steps
                    results.append(f"Step: {step}\nResult: {content}")
            
            # Build new state without clobbering last_sql/answer for non-DB turns
            # if there were no inference steps, dont overwrite last_sql/answer with None
            new_state = {
                **state,
                "db_results": "\n\n".join(results) if results else "No results were generated.",
            }

            if saw_inference:
                new_state["last_sql"] = last_sql
                new_state["last_answer"] = last_answer
            else:
                # Optional: if this turn was pure chit-chat, you might prefer clearing these:
                new_state["last_sql"] = None
                new_state["last_answer"] = None
                pass

            return new_state

        except Exception as e:
            logger.error(f"Error in execute_plan: {str(e)}", exc_info=True)
            return {**state, "db_results": f"Error executing steps: {str(e)}",
                    "last_sql": last_sql,
                    "last_answer": last_answer}

    def generate_response(self, state: ConversationState) -> ConversationState:
        # Generate the final response based on the input type
        logger.info("Generating final response")
    
        # Normal response generation for non-clarification cases
        is_chat = state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]

        # Keep the same language as the user's question
        lang = state.get("lang") or detect_lang(state.get("question", ""))
        response_language = "Portuguese (Portugal)" if lang == "pt" else "English"

        user_id = state["user_id"]
        long_term_mem = get_long_term_memory(user_id)

        vars = {
            "question": state.get("question", ""),
            "db_results": state.get("db_results", ""),
            "response_language": response_language,
            "last_sql": state.get("last_sql", "") or "",
            "last_answer": state.get("last_answer", "") or "",
            "candidates": ", ".join(state.get("candidates", []) or []),
            "long_term_memory": long_term_mem
        }

        wrapper = self.chat_with_mem if is_chat else self.db_with_mem

        llm_response = wrapper.invoke(
            vars,
            config={"configurable": {"session_id": user_id}}
        )

        content = llm_response.content or ""

        update_long_term_memory(user_id, vars["question"], content)

        # BROADEST APPROACH: If we're asking any question and not already in clarification mode,
        # treat the next user response as clarification        
        # response = state.get("response", "") or ""  # Ensure it's never None

        # Detect clarification AFTER we have the LLM content
        # language-aware confirmation triggers (small set)
        clarify_triggers_en = ["did you mean", "reply", "please confirm", "confirm", "reply with the exact event title", "choose", "verify", "pick", "select"]
        clarify_triggers_pt = ["queria dizer", "quer dizer", "responda", "responda com o título", "confirme", "poderia confirmar", "confirma", "confirmar", "verificar", "escolher", "escolha", "selecionar", "selecione"]

        lc = content.lower() if content else ""
        triggers = clarify_triggers_pt if lang == "pt" else clarify_triggers_en

        is_asking_question = (
            ('?' in content if content else False) or
            any(tok in lc for tok in triggers)
        ) and not state.get("awaiting_clarification", False)

        # is_asking_question = (
        #     content and 
        #     '?' in content and
        #     not state.get("awaiting_clarification", False)
        # )
        
        if is_asking_question:
            logger.info("Detected clarification question - setting awaiting_clarification=True")

            return {
                **state,
                "response": content,
                "awaiting_clarification": True,
                "clarification_question": content,
                "original_question": state.get("question")
            }

        # Update state with the response and clear the plan
        return {**state, "response": content, "plan": []}