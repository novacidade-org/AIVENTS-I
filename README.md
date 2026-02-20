# AIVENTS Chatbot - Model I

## Description

This repository contains AIVENTS, a multi-agent conversational AI system for assisting in events management. 

## Supported Python Version

- Recommended: Python 3.10.12
- Check your current Python version with:

```bash
python --version
```

## Create a Virtual Environment

It is recommended to use a virtual environment to control dependencies.

Using built-in venv:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows (PowerShell): .\.venv\Scripts\Activate.ps1
```

## Install Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

If you use `pip` from a system Python, consider upgrading pip first:

```bash
pip install --upgrade pip
```

## How to Run

- Run the main CLI/agent runner:

```bash
python main.py
```

- Run the Streamlit demo app (web UI):

```bash
streamlit run streamlit_app.py
```

## Project Layout (key files)

- `main.py`: Primary entry point for running the system.
- `streamlit_app.py`: Streamlit-based demo UI.
- `inference_agent.py`, `planner_agent.py`, `supervisor_agent.py`: Agent modules.
- `conversation_state.py`, `utils.py`, `logger.py`: Utilities and helpers.
- `requirements.txt`: Python dependencies.

## Notes

- Follow the virtual environment steps before installing dependencies.

<!-- ## Contributing

Open an issue or submit a pull request with improvements or bug fixes.

## License

Specify the project license here if any. -->
