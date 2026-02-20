import networkx as nx
from sqlalchemy import inspect
from config import Config
from logger import logger

class DiscoveryAgent:
    def __init__(self):
        # Initialize configuration and toolkit
        self.config = Config()
        
    def run_query(self, q):
        # Execute a SQL query using the configured database engine
        return self.config.sql_db.run(q)

    def define_glossary_relations(self, G: nx.Graph, tables: list[str], canonicalColumns: dict) -> None:
        """
        Read structured relations from self.config.glossary['tables'][table]['relations']
        and add column↔column edges to G when both sides exist in the current graph.
        """

        # Prefer normalized relations from table_meta if available; else fall back to raw glossary.
        use_meta = hasattr(self.config, "table_meta") and isinstance(self.config.table_meta, dict)
        table_meta = self.config.table_meta if use_meta else {}
        glossary_tables = (self.config.glossary or {}).get("tables", {})

        added = 0
        for src_table in tables:
            # pull relations list from table_meta first, otherwise glossary
            rels = (table_meta.get(src_table, {}) or {}).get("relations", [])
            if not rels:
                rels = (glossary_tables.get(src_table, {}) or {}).get("relations", []) or []

            for r in rels:
                if not isinstance(r, dict):
                    continue
                from_col  = r.get("from_col")
                to_table  = r.get("to_table")
                to_col    = r.get("to_col")
                kind      = r.get("kind", "fk")

                # validate and respect current graph scope
                if not (from_col and to_table and to_col):
                    continue
                if to_table not in tables:
                    # relation points to a table you didn't include in this graph
                    continue

                # resolve column nodes already created in `discover`
                src = canonicalColumns.get(f"{src_table}.{from_col}")
                dst = canonicalColumns.get(f"{to_table}.{to_col}")

                if src and dst:
                    G.add_edge(
                        src, dst,
                        kind="fk_glossary",
                        relation_kind=kind,
                        source="table_meta" if use_meta else "glossary",
                        from_table=src_table, from_col=from_col,
                        to_table=to_table,   to_col=to_col
                    )
                    added += 1
                else:
                    # Optional: debug logs if something doesn't line up
                    if not src:
                        logger.warning(f"[discover] Missing column node: {src_table}.{from_col}")
                    if not dst:
                        logger.warning(f"[discover] Missing column node: {to_table}.{to_col}")

        if added == 0:
            logger.info("[discover] No glossary relations added (none matched current tables/columns).")
            
    def discover(self, include_tables: list[str] = None) -> nx.Graph:
        """Use the real schema from SQLDatabase to construct the graph."""

        inspector = inspect(self.config.sql_db._engine)
        all_tables = inspector.get_table_names()

         # Restrict to selected tables if provided
        if include_tables:
            tables = [t for t in all_tables if t in include_tables]
        else:
            tables = all_tables
            
        G = nx.Graph()

        node_id = 0
        column_id = 1000
        canonicalColumns = {}

        for table in tables:
            node_id += 1
            G.add_node(node_id, tableName=table)

            columns = inspector.get_columns(table)
            for col in columns:
                column_id += 1
                colname = col["name"]
                coltype = str(col["type"])
                nullable = col.get("nullable", True)
                G.add_node(column_id, columnName=colname, columnType=coltype, isOptional=nullable)
                G.add_edge(node_id, column_id)
                canonicalColumns[f"{table}.{colname}"] = column_id

        # wire the relations from your glossary into this same graph
        self.define_glossary_relations(G, tables, canonicalColumns)    

        return G