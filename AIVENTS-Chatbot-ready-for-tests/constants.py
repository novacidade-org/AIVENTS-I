# Filter tables
selected_tables = ["silver_fact_transacoes_sibs", "silver_dim_transacoes_origem_cartao", "silver_dim_transacoes_setor", "silver_events_oeste", 
                   "silver_dim_data", "silver_dim_cim_concelho_freguesia"]

# se a pergunta tiver algum termo relacionado a eventos, forçar a tabela dos eventos a estar nas candidatas
event_table = "silver_events_oeste"

event_triggers = {
    "event", "events", "evento", "eventos",
    "festival", "festivals", "festivais",
    "fair", "fairs", "feira", "feiras",
    "party", "parties", "festa", "festas",
    "activity", "activities", "atividade", "atividades"
}