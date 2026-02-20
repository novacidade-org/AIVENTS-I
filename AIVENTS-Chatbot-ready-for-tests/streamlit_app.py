import streamlit as st
from main import graph  # import the already-created global graph
import uuid
import time

# ---------------------------
# Initialize chatbot session
# ---------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Persistent conversation state for LangGraph
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {"user_id": st.session_state.session_id}

# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AIVENTS Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AIVENTS Agent")
st.caption("Ask questions about your event & municipality database")


# ---------------------------
# Sidebar with memory + reset
# ---------------------------
with st.sidebar:
    st.header("🗂 Session")
    st.write(f"Session ID: `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        # Reset messages and state
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.conversation_state = {"user_id": st.session_state.session_id}
        st.rerun()
        # st.experimental_rerun()

    # NEW: Toggle to show the latest SQL
    show_sql = st.checkbox("Show last SQL query", value=False)

    if show_sql:
        current_state = st.session_state.conversation_state
        last_sql = current_state.get("last_sql")

        if last_sql:
            st.code(last_sql, language="sql")
        else:
            st.info("No SQL query generated yet.")

# ---------------------------
# Display chat history
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# New helper to stream text with a slight delay
def typewriter_text(text: str, delay: float = 0.025):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# ---------------------------
# User input handling
# ---------------------------
user_input = st.chat_input("Ask me something about your data...")

if user_input:

    # Save & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Inject this turn's question into the LangGraph state
    st.session_state.conversation_state["question"] = user_input

    # Invoke the graph (MAIN FIX)
    result = graph.invoke(st.session_state.conversation_state)

    # Persist returned state for next turn
    st.session_state.conversation_state = result

    # Extract bot response
    bot_response = result.get("response", "⚠️ No response generated.")

    # Display bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(typewriter_text(bot_response))

# import streamlit as st
# from main import graph
# from conversation_state import ConversationState
# import uuid

# # ---------------------------
# # Initialize chatbot and session
# # ---------------------------
# if "graph" not in st.session_state:
#     st.session_state.graph = create_graph()

# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# # ---------------------------
# # Streamlit Page Configuration
# # ---------------------------
# st.set_page_config(
#     page_title="AIVENTS Agent",
#     page_icon="🤖",
#     layout="wide"
# )

# st.title("🤖 AIVENTS Agent")
# st.caption("Ask questions about your event & municipality database")


# # ---------------------------
# # Sidebar with memory + reset
# # ---------------------------
# with st.sidebar:
#     st.header("🗂 Session")
#     st.write(f"Session ID: `{st.session_state.session_id}`")

#     if st.button("🔄 Reset Conversation"):
#         st.session_state.messages = []
#         st.session_state.session_id = str(uuid.uuid4())
#         st.session_state.graph = create_graph()
#         st.experimental_rerun()


# # ---------------------------
# # Display chat history
# # ---------------------------
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])


# # ---------------------------
# # User input handling
# # ---------------------------
# user_input = st.chat_input("Ask me something about your data...")

# if user_input:
#     # Initialize persistent conversation state
#     if "conversation_state" not in st.session_state:
#         st.session_state.conversation_state = {"user_id": st.session_state.session_id}

#     # Inject user question into that state
#     st.session_state.conversation_state["question"] = user_input

#     # Invoke graph with the persistent state
#     result = graph.invoke(st.session_state.conversation_state)
#     st.session_state.conversation_state = result

#     # Display user message
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Call your LangGraph pipeline
#     graph = st.session_state.graph

#     result = graph.invoke({
#         "user_id": st.session_state.session_id,
#         "question": user_input
#     })

#     bot_response = result.get("response", "⚠️ No response generated.")

#     # Display bot response
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

#     with st.chat_message("assistant"):
#         st.write(bot_response)

# import uuid
# import streamlit as st

# from main import create_graph


# st.set_page_config(page_title="AIVENTS Chatbot", layout="wide")


# def init():
#     if 'graph' not in st.session_state:
#         st.session_state.graph = create_graph()
#     if 'session_id' not in st.session_state:
#         st.session_state.session_id = str(uuid.uuid4())
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []


# def send_message(text: str):
#     # Append user message
#     st.session_state.messages.append({"role": "user", "text": text})

#     # Invoke the graph with a simple state (main.py expects user_id + question)
#     graph = st.session_state.graph
#     try:
#         state = graph.invoke({
#             "user_id": st.session_state.session_id,
#             "question": text,
#         })
#         bot_response = state.get("response", "[No response]")
#     except Exception as e:
#         bot_response = f"[Error invoking chatbot: {e}]"

#     st.session_state.messages.append({"role": "assistant", "text": str(bot_response)})


# def clear_conversation():
#     st.session_state.messages = []


# def render_messages():
#     for msg in st.session_state.messages:
#         if msg['role'] == 'user':
#             st.markdown(f"**You:** {msg['text']}")
#         else:
#             st.markdown(f"**Bot:** {msg['text']}")


# def main():
#     init()

#     with st.sidebar:
#         st.title("AIVENTS — Chatbot")
#         st.markdown("Simple Streamlit UI for the chatbot pipeline.")
#         if st.button("Clear conversation"):
#             clear_conversation()
#         st.markdown("---")
#         st.markdown("Run locally: `streamlit run streamlit_app.py`")

#     st.title("AIVENTS Chatbot")

#     # Messages area
#     chat_area = st.container()

#     with chat_area:
#         render_messages()

#     # Input area
#     col1, col2 = st.columns([8, 1])
#     with col1:
#         user_input = st.text_input("Your message", key="user_input")
#     with col2:
#         send = st.button("Send")

#     if send and user_input:
#         send_message(user_input)
#         # Clear input after send (works around streamlit state)
#         st.session_state.user_input = ""


# if __name__ == '__main__':
#     main()
