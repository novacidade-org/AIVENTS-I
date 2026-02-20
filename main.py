# Imports
from langgraph.graph import StateGraph, START, END
from logger import logger
from utils import detect_lang
from constants import selected_tables
from config import Config
from discovery_agent import DiscoveryAgent
from conversation_state import ConversationState
from supervisor_agent import SupervisorAgent

# --------------------------------------------------
# GLOBAL CONFIG AND GLOBAL DB GRAPH (load at startup)
# --------------------------------------------------

global_config = Config()
# Create global discovery graph ONCE at startup
global_db_graph = DiscoveryAgent().discover(include_tables=selected_tables)

def process_clarification(state: ConversationState) -> ConversationState:
    """Process user's clarification response and update the question accordingly."""
    q = state["question"]
    original_question = state.get("original_question", "")
    clarification_question = state.get("clarification_question", "")
    
    logger.info(f"Processing clarification: original='{original_question}', clarification_response='{q}'")
    
    # Simple rule-based clarification processing for location questions
    response_lower = q.lower().strip()
    
    # Check for location type clarifications
    if any(term in clarification_question.lower() for term in ["parish", "municipality", "freguesia", "concelho"]):
        if any(word in response_lower for word in ["parish", "freguesia"]):
            resolved_question = f"{original_question} (parish)"
            logger.info(f"Resolved as parish: {resolved_question}")
        elif any(word in response_lower for word in ["municipality", "concelho"]):
            resolved_question = f"{original_question} (municipality)" 
            logger.info(f"Resolved as municipality: {resolved_question}")
        else:
            # Default to parish for simple responses like "parish"
            resolved_question = f"{original_question} (parish)"
            logger.info(f"Defaulting to parish: {resolved_question}")
    else:
        # For other types of clarification, just append the response
        resolved_question = f"{original_question} [clarification: {q}]"
        logger.info(f"Generic clarification: {resolved_question}")
    
    # Clear clarification state and prepare for database query
    return {
        **state,
        "question": resolved_question,
        "input_type": "DATABASE_QUERY",
        "awaiting_clarification": False,
        "original_question": None,
        "clarification_question": None,
        # Clear previous results to force fresh processing
        "plan": [],
        "db_results": None,
        "response": None
    }

# def check_valid_clarification_response(response: str, clarification_question: str) -> bool:
#     """Check if the user's response provides enough information to proceed."""
#     config = Config()
    
#     # Use LLM to determine if this is a valid clarification response
#     system_prompt = """You are a clarification validator. Determine if the user's response properly answers the clarification question.
    
#     A valid clarification response:
#     - Directly answers the question asked
#     - Provides specific information requested
#     - Is not another question
#     - Is not ambiguous or vague
#     - Has enough substance to proceed
    
#     An invalid clarification response:
#     - Asks another question instead of answering
#     - Is too vague (e.g., "maybe", "I don't know", "not sure")
#     - Requests more information instead of providing it
#     - Is off-topic
#     - Is just "yes" or "no" without specifying which option
    
#     Respond with ONLY "VALID" or "INVALID"."""

#     messages = [
#         ("system", system_prompt),
#         ("human", f"Clarification question: {clarification_question}\nUser response: {response}")
#     ]
    
#     try:
#         llm_response = config.llm_classifier.invoke(messages)
#         validation = llm_response.content.strip().upper()
        
#         logger.info(f"Clarification validation: '{response}' -> {validation} for question: '{clarification_question}'")
        
#         return validation == "VALID"
        
#     except Exception as e:
#         logger.error(f"Error validating clarification: {e}")
#         # Fallback: basic validation
#         return basic_fallback_validation(response, clarification_question)

def basic_fallback_validation(response: str, clarification_question: str) -> bool:
    """Fallback validation if LLM fails."""
    response_lower = response.lower().strip()
    clarification_lower = clarification_question.lower()
    
    # Invalid patterns
    invalid_patterns = [
        # Questions
        response_lower.endswith('?'),
        any(word in response_lower for word in ["what", "when", "where", "how", "why", "which", "can you", "could you"]),
        
        # Vague responses
        response_lower in ["maybe", "i don't know", "not sure", "unsure", "idk", "not certain"],
        
        # Requests for more info
        any(phrase in response_lower for phrase in ["can you explain", "what do you mean", "i need more", "please clarify"]),
        
        # Too short and vague
        len(response.strip()) < 2,
        response_lower in ["yes", "no", "ok", "k", "y", "n"]
    ]
    
    # If any invalid pattern matches, it's invalid
    if any(invalid_patterns):
        return False
    
    # Otherwise, assume it's valid if it has some substance
    return len(response.strip()) > 1

# def ask_followup_clarification(state: ConversationState) -> ConversationState:
#     """Ask a follow-up clarification question when the initial response wasn't clear enough."""
#     q = state["question"]
#     original_question = state.get("original_question", "")
#     clarification_question = state.get("clarification_question", "")
#     lang = detect_lang(original_question)
    
#     # Determine what kind of follow-up clarification is needed
#     clarification_lower = clarification_question.lower()
    
#     if any(term in clarification_lower for term in ["parish", "municipality", "freguesia", "concelho"]):
#         # Location clarification wasn't clear
#         if lang == "pt":
#             followup = "Desculpe, não ficou claro. Quer dizer a freguesia ou o concelho? Por favor responda com 'freguesia' ou 'concelho'."
#         else:
#             followup = "I'm sorry, that wasn't clear. Do you mean the parish or the municipality? Please respond with 'parish' or 'municipality'."
    
#     elif any(term in clarification_lower for term in ["event", "evento", "title", "título"]):
#         # Event title clarification wasn't clear
#         if lang == "pt":
#             followup = "Por favor, copie e cole o título exato do evento da lista acima."
#         else:
#             followup = "Please copy and paste the exact event title from the list above."
    
#     else:
#         # Generic follow-up
#         if lang == "pt":
#             followup = "Pode clarificar melhor a sua resposta?"
#         else:
#             followup = "Could you clarify your response further?"
    
#     return {
#         **state,
#         "response": followup,
#         "awaiting_clarification": True,  # Still awaiting clarification
#         # Keep original_question and clarification_question for the next round
#     }

def classify_user_input(state: ConversationState) -> ConversationState:
    """Classifies user input to determine if it requires database access."""
    q = state["question"]
    lang = detect_lang(q)
    config = global_config

    # SIMPLE RULE: If we're awaiting clarification, ANY response is a clarification response
    if state.get("awaiting_clarification"):
        logger.info(f"Processing clarification response: '{q}' for original question: '{state.get('original_question')}'")
        
        # This is definitely a clarification response - route to processing
        return {
            **state,
            "input_type": "CLARIFICATION_RESPONSE",
            "lang": lang
        }

    # Normal classification for new questions
    system_prompt = """You are an input classifier. Classify the user's input into one of these categories:
    - DATABASE_QUERY: Questions about data, requiring database access
    - GREETING: General greetings, how are you, etc.
    - CHITCHAT: General conversation not requiring database
    - FAREWELL: Goodbye messages

    Respond with ONLY the category name."""

    messages = [
        ("system", system_prompt),
        ("user", state['question'])
    ]
    
    response = config.llm_classifier.invoke(messages)
    classification = response.content.strip()

    logger.info(f"Input classified as: {classification}")

    return {
        **state,
        "input_type": classification,
        "lang": lang,
        # Initialize clarification state for new questions
        "awaiting_clarification": False,
        "original_question": None,
        "clarification_question": None
    }

def route_after_classification(state: ConversationState) -> str:
    """Route after input classification."""
    if state.get("input_type") == "CLARIFICATION_RESPONSE":
        return "process_clarification"
    elif state.get("input_type") == "DATABASE_QUERY":
        return "discover_database"
    else:
        return "generate_response"

def route_after_clarification(state: ConversationState) -> str:
    """Route after processing clarification response."""
    # After processing clarification, we should have a resolved question
    # that can be treated as a DATABASE_QUERY
    if state.get("awaiting_clarification"):
        # Still need more clarification - ask follow-up
        return "generate_response"
    else:
        # Got valid clarification - proceed with database query
        return "discover_database"

def discover_database(state: ConversationState) -> ConversationState:
    """Inject the preloaded DB graph (already discovered at startup)."""
    if state.get("db_graph") is None:
        logger.info("Using cached DB schema (startup discovery).")
        return {**state, "db_graph": global_db_graph}
    return state
    # # Check if the database graph is already present in the state
    # if state.get('db_graph') is None:
    #     logger.info("Performing one-time database schema discovery...")
        
    #     # Use the DiscoveryAgent to generate the database graph
    #     discovery_agent = DiscoveryAgent()
    #     graph = discovery_agent.discover(include_tables=selected_tables)
        
    #     logger.info("Database schema discovery complete - this will be reused for future queries")
        
    #     # Update the state with the discovered database graph
    #     return {**state, "db_graph": graph}
    
    # # Return the existing state if the database graph already exists
    # return state

def create_graph():
    # Initialize the supervisor agent and state graph builder
    supervisor = SupervisorAgent()
    builder = StateGraph(ConversationState)

    # Add nodes representing processing steps in the flow
    builder.add_node("classify_input", classify_user_input)  # Classify the user input
    builder.add_node("discover_database", discover_database)  # Perform database discovery
    builder.add_node("create_plan", supervisor.create_plan)  # Create a plan based on input
    builder.add_node("execute_plan", supervisor.execute_plan)  # Execute the generated plan
    builder.add_node("generate_response", supervisor.generate_response)  # Generate the final response

    # Add clarification processing node
    builder.add_node("process_clarification", process_clarification)

    # Define the flow of states
    builder.add_edge(START, "classify_input")  # Start with input classification

    # After classification, route based on input type and clarification status
    builder.add_conditional_edges(
        "classify_input",
        route_after_classification,  # Use the routing function that checks for clarification
        path_map={
            "process_clarification": "process_clarification",  # Route clarification responses to processing
            "discover_database": "discover_database",
            "generate_response": "generate_response"
        }
    )

    # From process_clarification, decide whether to continue with database query or ask more clarification
    builder.add_conditional_edges(
        "process_clarification",
        route_after_clarification,  # Use the new routing function
        path_map={
            "discover_database": "discover_database",  # Valid clarification - proceed to DB
            "generate_response": "generate_response"    # Need more clarification - ask follow-up
        }
    )

    # From discovery to planning
    builder.add_edge("discover_database", "create_plan")
    
    # Conditionally execute the plan or generate a response if no plan exists
    builder.add_conditional_edges(
        "create_plan",
        lambda x: "execute_plan" if x.get("plan") else "generate_response",
        path_map={
            "execute_plan": "execute_plan",
            "generate_response": "generate_response"
        }
    )

    # Connect execution to response generation
    builder.add_edge("execute_plan", "generate_response")

    # End the process after generating the response
    builder.add_edge("generate_response", END)

    # Compile and return the state graph
    return builder.compile()

# Create the graph for processing
graph = create_graph()

def run_console_chat():
    # graph = create_graph()
    print("Chatbot ready! Type 'exit' to quit.\n")
    
    session_state = {
        "user_id": "0",
        "question": None,
        # Anything else you want to persist across turns
    }

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        # update state with new question
        session_state["question"] = user_input

        # run graph
        new_state = graph.invoke(session_state)

        # persist returned state for next turn
        session_state = new_state

        logger.info("SESSION STATE GUARDADO")
        logger.info(session_state)
        
        # state = graph.invoke({
        #     "user_id": session_id,
        #     "question": user_input,
        # })

        print("Bot:", new_state.get("response", "[No response]"))

if __name__ == "__main__":
    run_console_chat()