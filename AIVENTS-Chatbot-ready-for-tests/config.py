import os
import yaml
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from azure.identity import EnvironmentCredential
from langchain_openai import AzureChatOpenAI
from itertools import chain, repeat
import struct
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pyodbc
from sqlalchemy.pool import QueuePool
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
from constants import selected_tables

load_dotenv()

def load_glossary(path: str = "glossary_tables.yaml") -> dict:
    """Read and lightly validate the schema glossary (YAML → dict)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) # convert the YAML into a Python object
    if not isinstance(data, dict) or "tables" not in data:
        raise ValueError("Glossary must be a dict with a top-level 'tables' key.")
    # light sanity check for columns
    for t, meta in data["tables"].items():
      if "columns" in meta:
        cols = meta["columns"]
        if len(cols) != len(set(cols.keys())):
            raise ValueError(f"Duplicate column names in {t}.")
    return data # as a Python dict

def build_table_meta(glossary: dict) -> dict:
    """Table-level titles/synonyms/content for prompt 'schema cards' and offline matching."""
    meta = {}
    for t, m in glossary["tables"].items():
        rels_in = m.get("relations") or []
        relations = []
        for r in rels_in:
            if isinstance(r, dict) and r.get("from_col") and r.get("to_table") and r.get("to_col"):
                relations.append({
                    "from_col": r["from_col"],
                    "to_table": r["to_table"],
                    "to_col":   r["to_col"],
                    "kind":     r.get("kind", "fk"),
                })

        meta[t] = {
            "title_pt": m.get("title_pt", ""),
            "title_en": m.get("title_en", ""),
            "synonyms_pt":   m.get("synonyms_pt", []),
            "synonyms_en":   m.get("synonyms_en", []),
            "content_pt":  m.get("content_pt", ""),
            "content_en":  m.get("content_en", ""),
            "relations":  relations,
        }
    return meta

def build_column_catalog(glossary: dict) -> dict:
    """Column-level labels/synonyms for ColumnLinker."""
    catalog = {}
    for t, meta in glossary["tables"].items():
        cols = (meta or {}).get("columns") or {}
        catalog[t] = {}
        for c, labels in cols.items():
            catalog[t][c] = {
                "pt": labels.get("pt", ""),
                "en": labels.get("en", ""),
                "synonyms_pt": labels.get("synonyms_pt", []),
                "synonyms_en": labels.get("synonyms_en", []),
            }
    return catalog

def filter_glossary(full_glossary: dict, selected_tables: List[str], *, strict: bool = False) -> dict:
    """Return a glossary that contains ONLY entries for selected_tables.
    If strict=True, raise if any selected table is missing from the glossary."""
    missing = [t for t in selected_tables if t not in full_glossary["tables"]]
    if missing and strict:
        raise ValueError(f"Selected tables missing in glossary: {missing}")

    return {
        "tables": {t: full_glossary["tables"][t]
                   for t in selected_tables
                   if t in full_glossary["tables"]}
    }


# setup centralised config object to keep our code DRY
class Config:
    cached_credential = None
    def __init__(self):
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_KEY")    # os.getenv("AZURE_OPENAI_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")   # os.getenv("AZURE_OPENAI_ENDPOINT")
        self.fabric_sql_endpoint = os.getenv("FABRIC_SQL_ENDPOINT")     # os.getenv("FABRIC_SQL_ENDPOINT")
        self.fabric_database = os.getenv("FABRIC_DATABASE")     # os.getenv("FABRIC_DATABASE")
        self.azure_tenant_id = os.getenv("AZURE_TENANT_ID")
        self.azure_client_id = os.getenv("AZURE_CLIENT_ID")
        self.azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
       
        # Set environment variables for EnvironmentCredential
        os.environ['AZURE_TENANT_ID'] = self.azure_tenant_id
        os.environ['AZURE_CLIENT_ID'] = self.azure_client_id
        os.environ['AZURE_CLIENT_SECRET'] = self.azure_client_secret
        
        # self.credential = EnvironmentCredential()
        if Config.cached_credential is None:
            Config.cached_credential = EnvironmentCredential()
            
        self.credential = EnvironmentCredential()

        if not all([
            self.azure_openai_api_key,
            self.azure_openai_endpoint,
            self.fabric_sql_endpoint,
            self.fabric_database,
            self.azure_tenant_id,
            self.azure_client_id,
            self.azure_client_secret
        ]):
            raise ValueError("Missing required environment variables")

        # LLM setup
        self.llm_inference = AzureChatOpenAI(
            azure_deployment="gpt-4o-2",
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2024-12-01-preview",
            api_key=self.azure_openai_api_key,
            temperature=1,
        )

        self.llm_planner = AzureChatOpenAI(
            azure_deployment="gpt-4o-2",
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2024-12-01-preview",
            api_key=self.azure_openai_api_key,
            temperature=0,
        )

        self.llm_supervisor = AzureChatOpenAI(
            azure_deployment="gpt-4o-2",
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2024-12-01-preview",
            api_key=self.azure_openai_api_key,
            temperature=0,
        )

        self.llm_classifier = AzureChatOpenAI(
            azure_deployment="gpt-4o-2",
            azure_endpoint=self.azure_openai_endpoint,
            api_version="2024-12-01-preview",
            api_key=self.azure_openai_api_key,
            temperature=1,
        )

        def _connect_with_fresh_token():
            token = self.credential.get_token("https://database.windows.net/.default")
            token_as_bytes = bytes(token.token, "UTF-8")
            encoded = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
            token_struct = struct.pack("<i", len(encoded)) + encoded

            odbc_str = (
                f"Driver={{ODBC Driver 18 for SQL Server}};"
                f"Server={self.fabric_sql_endpoint},1433;"
                f"Database={self.fabric_database};"
                "Encrypt=Yes;TrustServerCertificate=No;"
            )
            return pyodbc.connect(odbc_str, attrs_before={1256: token_struct})
        
        # Add retry decorator
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((pyodbc.OperationalError, pyodbc.InterfaceError))
        )
        def _connect_with_retry():
            return _connect_with_fresh_token()

        self.db_engine = create_engine(
            "mssql+pyodbc://",
            creator=_connect_with_retry,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=3000
        )

        # GLOSSARY
        # Load full YAML then filter down to only the selected tables
        full_glossary = load_glossary("glossary_tables.yaml")
        self.selected_tables = selected_tables
        self.glossary = filter_glossary(full_glossary, self.selected_tables, strict=True)

        missing = [t for t in self.selected_tables if t not in self.glossary["tables"]]
        if missing:
            print(f"[Config] Warning: no glossary entry for: {missing} (they'll still be queryable, "
                  "but won’t benefit from schema linking/cards)")
        
        # Derived, kept in sync with helpers
        self.table_names = list(self.glossary["tables"].keys())
        self.table_meta = build_table_meta(self.glossary)
        self.column_catalog = build_column_catalog(self.glossary)

        # self.sql_db = SQLDatabase(engine=self.db_engine)
        self.sql_db = SQLDatabase(engine=self.db_engine,
                                  include_tables=self.selected_tables,
                                  lazy_table_reflection=True)