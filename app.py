import os
import re
import io
import hashlib
import sqlite3

import streamlit as st

from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

if "active_db" not in st.session_state:
    st.session_state.active_db = None

if "active_retriever" not in st.session_state:
    st.session_state.active_retriever = None

if "active_source" not in st.session_state:
    st.session_state.active_source = "Demo Dataset"

if "data_mode" not in st.session_state:
    st.session_state.data_mode = "demo"

if "csv_hash" not in st.session_state:
    st.session_state.csv_hash = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "relationship_context" not in st.session_state:
    st.session_state.relationship_context = ""

if "relationship_count" not in st.session_state:
    st.session_state.relationship_count = 0

if "demo_hash" not in st.session_state:
    st.session_state.demo_hash = None


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

# PAGE CONFIGURATION


st.set_page_config(
    page_title="Text-to-SQL AI Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM UI STYLING

st.markdown(
    """
    <style>
    :root { --ink:#17152b; --line:#e8e5f3; --purple:#7c3aed; --violet:#8b5cf6; --pink:#ec4899; --cyan:#06b6d4; }
    .stApp { background: radial-gradient(circle at 78% 8%, rgba(124,58,237,.08), transparent 23%), radial-gradient(circle at 8% 65%, rgba(6,182,212,.045), transparent 20%), #f8f7fc; }
    .block-container { max-width:1280px; padding-top:1.35rem; padding-bottom:4.5rem; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#151329 0%,#1b1738 52%,#17152b 100%); border-right:1px solid rgba(255,255,255,.07); }
    section[data-testid="stSidebar"] .block-container { padding:1.05rem .95rem 2rem; }
    section[data-testid="stSidebar"] * { color:#e9e7f7; }
    section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.10); }
    /* CSV UPLOADER — keep every uploader state on-brand, including the uploaded-file row */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] { background:linear-gradient(145deg,rgba(124,58,237,.13),rgba(236,72,153,.07)); border:1px solid rgba(196,181,253,.20); border-radius:16px; padding:8px; box-shadow:inset 0 1px 0 rgba(255,255,255,.04); }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:linear-gradient(145deg,rgba(124,58,237,.16),rgba(91,33,182,.12)); border:1px dashed rgba(196,181,253,.48); border-radius:12px; min-height:112px; transition:all .18s ease; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover { background:linear-gradient(145deg,rgba(139,92,246,.25),rgba(236,72,153,.13)); border-color:rgba(244,114,182,.70); box-shadow:0 0 0 3px rgba(139,92,246,.08),0 10px 25px rgba(124,58,237,.12); }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] { color:#d9d5ed; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small { color:#9f9abf !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span { color:#eeeaff !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background:linear-gradient(135deg,#8b5cf6,#ec4899) !important; color:#fff !important; border:0 !important; border-radius:9px !important; box-shadow:0 6px 18px rgba(139,92,246,.24) !important; font-weight:750 !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover { background:linear-gradient(135deg,#7c3aed,#db2777) !important; color:#fff !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] { background:linear-gradient(135deg,rgba(139,92,246,.22),rgba(236,72,153,.12)) !important; border:1px solid rgba(196,181,253,.28) !important; border-radius:10px !important; color:#f5f3ff !important; box-shadow:0 5px 16px rgba(0,0,0,.12) !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * { color:#f5f3ff !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg { fill:#d8ccff !important; color:#d8ccff !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button { background:rgba(255,255,255,.08) !important; border:0 !important; color:#e9e7f7 !important; box-shadow:none !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button:hover { background:rgba(236,72,153,.20) !important; color:#fff !important; }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] [role="progressbar"] { background:linear-gradient(90deg,#8b5cf6,#ec4899,#06b6d4) !important; }
    section[data-testid="stSidebar"] div.stButton > button { background:rgba(255,255,255,.055); color:#e9e7f7; border:1px solid rgba(255,255,255,.09); box-shadow:none; }
    section[data-testid="stSidebar"] div.stButton > button:hover { background:linear-gradient(135deg,rgba(124,58,237,.30),rgba(236,72,153,.16)); border-color:rgba(196,181,253,.50); color:#fff; transform:translateY(-1px); }
    .sidebar-brand { padding:5px 4px 15px; }
    .brand-row { display:flex; align-items:center; gap:10px; }
    .brand-mark { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:18px; background:linear-gradient(135deg,#8b5cf6,#ec4899); box-shadow:0 8px 24px rgba(139,92,246,.32); }
    .sidebar-brand .brand-name { color:#fff; font-size:1rem; font-weight:800; letter-spacing:-.025em; }
    .sidebar-brand .brand-sub { color:#aaa6c5; font-size:.69rem; margin-top:2px; }
    .sidebar-section { color:#9f9abf; font-size:.64rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:4px 0 8px; }
    .source-card { background:linear-gradient(135deg,rgba(124,58,237,.20),rgba(236,72,153,.09)); border:1px solid rgba(196,181,253,.24); border-radius:15px; padding:13px 14px; margin:9px 0 3px; }
    .source-label { color:#aaa6c5; font-size:.59rem; font-weight:800; letter-spacing:.13em; }
    .source-name { color:#fff; font-size:.92rem; font-weight:800; margin-top:5px; }
    .source-detail { color:#b9b4d0; font-size:.68rem; margin-top:3px; overflow-wrap:anywhere; }

    /* HERO */
    .hero { position:relative; overflow:hidden; background:linear-gradient(135deg,#18152f 0%,#30206b 45%,#6747e9 100%); border:1px solid rgba(255,255,255,.08); border-radius:28px; padding:38px 42px 36px; margin-bottom:18px; box-shadow:0 24px 65px rgba(53,39,122,.20); }
    .hero:before { content:""; position:absolute; width:310px; height:310px; right:-80px; top:-130px; border-radius:50%; background:radial-gradient(circle,rgba(236,72,153,.38),rgba(236,72,153,0) 68%); }
    .hero:after { content:""; position:absolute; width:250px; height:250px; right:190px; bottom:-180px; border-radius:50%; background:radial-gradient(circle,rgba(6,182,212,.20),rgba(6,182,212,0) 70%); }
    .hero-content { position:relative; z-index:2; }
    .hero-eyebrow { display:inline-flex; align-items:center; gap:7px; color:#d8ccff; font-size:.64rem; font-weight:850; letter-spacing:.16em; margin-bottom:10px; }
    .hero-eyebrow:before { content:"✦"; color:#f0abfc; font-size:.8rem; }
    .hero-title { color:#fff !important; font-size:2.55rem; line-height:1.05; font-weight:850; letter-spacing:-.055em; margin:0; }
    .hero-text { color:#dedbed !important; font-size:.94rem; line-height:1.7; max-width:760px; margin:13px 0 0; }
    .hero-pill { display:inline-flex; align-items:center; margin-top:20px; padding:8px 13px; border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.08); color:#f5f3ff; border-radius:999px; font-size:.68rem; backdrop-filter:blur(10px); }

    /* CARDS */
    .info-card { position:relative; overflow:hidden; background:rgba(255,255,255,.88); border:1px solid #e8e5f3; border-radius:18px; padding:16px 17px; min-height:92px; box-shadow:0 8px 25px rgba(39,30,77,.055); transition:all .18s ease; }
    .info-card:before { content:""; position:absolute; left:0; top:0; width:100%; height:3px; background:linear-gradient(90deg,#7c3aed,#ec4899,#06b6d4); }
    .info-card:hover { transform:translateY(-2px); box-shadow:0 13px 32px rgba(39,30,77,.09); border-color:#d7d0f5; }
    .info-label { color:#89839e; font-size:.61rem; font-weight:850; letter-spacing:.09em; text-transform:uppercase; }
    .info-value { color:var(--ink); font-size:.98rem; font-weight:800; margin-top:7px; }
    .info-note { color:#9892aa; font-size:.68rem; margin-top:3px; }
    .section-kicker { color:#8b5cf6; font-size:.62rem; font-weight:850; letter-spacing:.14em; text-transform:uppercase; margin-bottom:2px; }
    .section-title { color:var(--ink); font-size:1.22rem; font-weight:850; letter-spacing:-.03em; margin-bottom:3px; }
    .section-subtitle { color:#777188; font-size:.77rem; margin-bottom:13px; }

    /* BUTTONS */
    div.stButton > button { border:1px solid #e2deef; border-radius:13px; background:rgba(255,255,255,.9); color:#403b55; min-height:44px; font-weight:600; transition:all .16s ease; box-shadow:0 3px 10px rgba(31,25,61,.035); }
    div.stButton > button:hover { border-color:#b8a9f7; color:#5b21b6; background:linear-gradient(135deg,#fbf9ff,#fff7fc); transform:translateY(-2px); box-shadow:0 8px 22px rgba(124,58,237,.10); }

    /* CHAT */
    div[data-testid="stChatMessage"] { border:1px solid #e9e5f1; border-radius:18px; padding:11px 15px; margin-bottom:11px; background:rgba(255,255,255,.90); box-shadow:0 5px 18px rgba(31,25,61,.035); }
    div[data-testid="stChatMessage"] p { color:#39344a; line-height:1.68; }
    div[data-testid="stChatInput"] { background:transparent; }
    div[data-testid="stChatInput"] textarea { border:1px solid #dcd6ed !important; border-radius:16px !important; background:#fff !important; box-shadow:0 8px 26px rgba(39,30,77,.07) !important; }
    div[data-testid="stChatInput"] textarea:focus { border-color:#8b5cf6 !important; box-shadow:0 0 0 3px rgba(139,92,246,.10),0 10px 30px rgba(39,30,77,.08) !important; }

    /* RESULTS */
    div[data-testid="stExpander"] { border:1px solid #e6e2ef; border-radius:13px; background:#fcfbfe; box-shadow:0 3px 12px rgba(31,25,61,.025); }
    pre { border-radius:11px !important; border:1px solid #e7e3ef !important; }
    div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #e6e2ef; }
    .footer { text-align:center; color:#9a95aa; font-size:.68rem; padding:16px 0 0; }
    @media (max-width:800px) { .hero{padding:28px 23px;border-radius:21px}.hero-title{font-size:1.9rem}.block-container{padding-top:.8rem} }
    </style>
    """,
    unsafe_allow_html=True
)


# DATABASE CONFIGURATION

username = os.getenv("MYSQL_USERNAME", "root")
password = os.getenv("MYSQL_PASSWORD")
host = os.getenv("MYSQL_HOST", "localhost")
port = os.getenv("MYSQL_PORT", "3306")
database_schema = os.getenv("MYSQL_DATABASE", "text_to_sql")

mysql_uri = (
    f"mysql+pymysql://{username}:{password}"
    f"@{host}:{port}/{database_schema}"
)

# DATABASE CONNECTION

@st.cache_resource
def get_database():
    return SQLDatabase.from_uri(
        mysql_uri,
        sample_rows_in_table_info=5
    )


def get_mysql_database():
    return get_database()


db = None


def _safe_table_name(filename, used_names=None):
    """Convert a CSV filename into a safe, readable SQLite table name."""
    used_names = used_names or set()
    base = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", base).strip("_").lower()
    if not name:
        name = "uploaded_data"
    if name[0].isdigit():
        name = f"table_{name}"

    original = name
    counter = 2
    while name in used_names:
        name = f"{original}_{counter}"
        counter += 1
    return name


def _database_path(file_hash, prefix="uploaded_csv"):
    """Return a writable SQLite path for the active uploaded dataset."""
    db_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        f"{prefix}_{file_hash}.db"
    )

    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        test_path = os.path.join(os.path.dirname(db_path), ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception:
        import tempfile
        db_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{file_hash}.db")

    return db_path


def _normalise_identifier(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _column_tokens(value):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", str(value).lower()) if t]


def _is_id_like(column_name):
    tokens = _column_tokens(column_name)
    normalized = _normalise_identifier(column_name)
    return (
        normalized in {"id", "index", "code", "key"}
        or any(t in {"id", "index", "code", "key"} for t in tokens)
        or normalized.endswith("id")
        or normalized.endswith("index")
        or normalized.endswith("code")
        or normalized.endswith("key")
    )


def _table_stems(table_name):
    normalized = _normalise_identifier(table_name)
    stems = {normalized}
    if normalized.endswith("ies"):
        stems.add(normalized[:-3] + "y")
    if normalized.endswith("s"):
        stems.add(normalized[:-1])
    return {s for s in stems if s}


def _value_set(series, limit=5000):
    """Create a normalized sample set for relationship detection."""
    values = series.dropna().astype(str).str.strip()
    if len(values) > limit:
        values = values.sample(limit, random_state=42)
    return set(_normalise_identifier(v) for v in values if str(v).strip())


def detect_relationships(dataframes):
    """Infer likely foreign-key relationships from column names and value overlap.

    Relationships are deliberately labelled as detected/likely rather than
    treated as guaranteed database constraints.
    """
    relationships = []
    table_names = list(dataframes.keys())

    for source_table in table_names:
        source_df = dataframes[source_table]
        for target_table in table_names:
            if source_table == target_table:
                continue
            target_df = dataframes[target_table]

            target_stems = _table_stems(target_table)
            for source_col in source_df.columns:
                source_tokens = set(_column_tokens(source_col))
                source_norm = _normalise_identifier(source_col)

                for target_col in target_df.columns:
                    target_norm = _normalise_identifier(target_col)
                    target_tokens = set(_column_tokens(target_col))

                    # A target column is more useful as a referenced key when
                    # it is unique. Empty/all-null columns are ignored.
                    target_values = target_df[target_col].dropna().astype(str).str.strip()
                    if target_values.empty:
                        continue
                    uniqueness = target_values.nunique(dropna=True) / max(len(target_values), 1)
                    if uniqueness < 0.95:
                        continue

                    name_score = 0
                    if source_norm == target_norm:
                        name_score = 3
                    elif source_norm.endswith(target_norm) or target_norm.endswith(source_norm):
                        name_score = 2
                    elif source_tokens & target_tokens:
                        name_score = 1

                    semantic_score = 0
                    for stem in target_stems:
                        if stem and stem in source_norm:
                            semantic_score = max(semantic_score, 2)
                        if stem and stem in source_tokens:
                            semantic_score = max(semantic_score, 2)

                    if _normalise_identifier(target_col) in {"id", "index", "code", "key"}:
                        semantic_score += 1

                    if not (name_score >= 2 or semantic_score >= 2):
                        continue

                    source_values = _value_set(source_df[source_col])
                    target_values_set = _value_set(target_df[target_col])
                    if not source_values or not target_values_set:
                        continue

                    overlap = len(source_values & target_values_set) / max(len(source_values), 1)

                    # Require a strong value match. Lower thresholds are allowed
                    # when column/table naming gives a strong semantic signal.
                    threshold = 0.80 if (name_score >= 2 or semantic_score >= 2) else 0.95
                    if overlap < threshold:
                        continue

                    # Prefer one direction: source/referencing table -> target/key table.
                    relationships.append({
                        "source_table": source_table,
                        "source_column": source_col,
                        "target_table": target_table,
                        "target_column": target_col,
                        "overlap": round(overlap, 3),
                    })

    # Deduplicate reciprocal/duplicate detections.
    unique = {}
    for rel in relationships:
        key = (
            rel["source_table"], rel["source_column"],
            rel["target_table"], rel["target_column"]
        )
        unique[key] = rel

    return list(unique.values())


def format_relationship_context(relationships):
    if not relationships:
        return "No strong cross-table relationships were automatically detected."

    lines = ["AUTOMATICALLY DETECTED TABLE RELATIONSHIPS:"]
    for rel in relationships:
        lines.append(
            f"- {rel['source_table']}.{rel['source_column']} -> "
            f"{rel['target_table']}.{rel['target_column']} "
            f"(value overlap: {rel['overlap']:.0%}; likely relationship, not a declared constraint)"
        )
    return "\n".join(lines)


def build_multi_csv_database(file_items, db_prefix="uploaded_csv"):
    """Build one SQLite database containing one table per CSV file.

    file_items is a list of (filename, csv_bytes) tuples.
    """
    if not file_items:
        raise ValueError("Please provide at least one CSV file.")

    import pandas as pd

    combined_hash = hashlib.md5()
    dataframes = {}
    used_names = set()

    for filename, csv_bytes in file_items:
        if not filename.lower().endswith(".csv"):
            raise ValueError(f"Unsupported file: {filename}. Please upload CSV files only.")
        if not csv_bytes:
            raise ValueError(f"{filename} is empty.")

        combined_hash.update(filename.encode("utf-8"))
        combined_hash.update(csv_bytes)

        df = pd.read_csv(io.BytesIO(csv_bytes))
        if df.empty:
            raise ValueError(f"{filename} does not contain any data rows.")

        # Drop completely empty columns and make duplicate column names unique.
        df = df.dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        if any(not c for c in df.columns):
            df.columns = [c if c else f"column_{i+1}" for i, c in enumerate(df.columns)]
        df = df.loc[:, ~df.columns.duplicated()]

        table_name = _safe_table_name(filename, used_names)
        used_names.add(table_name)
        dataframes[table_name] = df

    file_hash = combined_hash.hexdigest()
    db_path = _database_path(file_hash, prefix=db_prefix)

    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            for table_name, df in dataframes.items():
                df.to_sql(table_name, conn, if_exists="replace", index=False)
        finally:
            conn.close()

    relationships = detect_relationships(dataframes)
    relationship_context = format_relationship_context(relationships)

    database = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=5
    )

    return database, relationship_context, relationships


def build_csv_database(csv_bytes, table_name="uploaded_data"):
    """Backward-compatible single-CSV wrapper."""
    database, relationship_context, relationships = build_multi_csv_database(
        [(f"{table_name}.csv", csv_bytes)]
    )
    return database


# GROQ LLM

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=500
    )


llm = get_llm()


# CREATE RAG DOCUMENTS FROM DATABASE SCHEMA


def create_schema_documents(database, relationship_context=None):
    """Build dynamic schema, sample-value and relationship documents."""
    documents = []
    try:
        from sqlalchemy import inspect
        inspector = inspect(database._engine)
        tables = database.get_usable_table_names()
        for table in tables:
            columns = inspector.get_columns(table)
            column_text = "\n".join(
                f"- {c['name']} | type: {c['type']}" for c in columns
            )
            try:
                sample = database.run(f'SELECT * FROM "{table}" LIMIT 5')
            except Exception:
                sample = "Sample rows unavailable."
            documents.append(Document(
                page_content=(
                    f"Table: {table}\nColumns:\n{column_text}\n\n"
                    f"Sample rows:\n{sample}"
                ),
                metadata={"table_name": table, "source": "database_schema"}
            ))
            for c in columns:
                documents.append(Document(
                    page_content=(
                        f"Table: {table}\nColumn: {c['name']}\n"
                        f"Data type: {c['type']}"
                    ),
                    metadata={"table_name": table, "column_name": c['name'], "source": "database_schema"}
                ))
    except Exception:
        for table in database.get_usable_table_names():
            documents.append(Document(
                page_content=database.get_table_info([table]),
                metadata={"table_name": table, "source": "database_schema"}
            ))

    if relationship_context:
        documents.append(Document(
            page_content=relationship_context,
            metadata={"source": "relationship_detection"}
        ))

    return documents

# EMBEDDINGS


@st.cache_resource
def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


embeddings = get_embeddings()


# FAISS VECTOR STORE

def build_retriever(database, relationship_context=None):
    documents = create_schema_documents(database, relationship_context)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 6})

retriever = None

# ACTIVE DATABASE HELPERS

def get_default_database():
    active_db = st.session_state.get("active_db")
    if active_db is not None:
        return active_db

    # Public demo is the default data source. MySQL remains supported when
    # explicitly selected/configured.
    if st.session_state.get("data_mode") == "mysql":
        return get_mysql_database()

    return get_demo_database()


def get_demo_database():
    """Load every CSV inside the app's sample_data folder as SQLite tables."""
    # Resolve relative to app.py, not the terminal's current working directory.
    # This makes the demo work even when Streamlit is launched from another folder.
    app_dir = os.path.dirname(os.path.abspath(__file__))
    demo_dir = os.path.join(app_dir, "sample_data")

    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(
            f"Demo data folder not found next to app.py: {demo_dir}"
        )

    file_paths = sorted(
        os.path.join(demo_dir, name)
        for name in os.listdir(demo_dir)
        if name.lower().endswith(".csv")
    )

    if not file_paths:
        raise FileNotFoundError(
            f"No CSV files found in {demo_dir}. "
            "Place your 6 dataset CSV files inside sample_data."
        )

    file_items = []
    for path in file_paths:
        with open(path, "rb") as f:
            file_items.append((os.path.basename(path), f.read()))

    combined_hash = hashlib.md5()
    for filename, csv_bytes in file_items:
        combined_hash.update(filename.encode("utf-8"))
        combined_hash.update(csv_bytes)
    demo_hash = combined_hash.hexdigest()

    if st.session_state.get("demo_hash") != demo_hash or st.session_state.get("active_db") is None:
        demo_db, relationship_context, relationships = build_multi_csv_database(
            file_items,
            db_prefix="demo_csv"
        )
        demo_retriever = build_retriever(demo_db, relationship_context)
        st.session_state.active_db = demo_db
        st.session_state.active_retriever = demo_retriever
        st.session_state.active_source = "Demo Dataset"
        st.session_state.data_mode = "demo"
        st.session_state.csv_hash = demo_hash
        st.session_state.demo_hash = demo_hash
        st.session_state.relationship_context = relationship_context
        st.session_state.relationship_count = len(relationships)
        st.session_state.uploaded_files = [name for name, _ in file_items]

    return st.session_state.active_db

def get_default_retriever(database):
    if st.session_state.get("mysql_retriever") is None:
        st.session_state.mysql_retriever = build_retriever(database)
    return st.session_state.mysql_retriever

# GET RAG CONTEXT

def get_rag_context(question):
    """Retrieve relevant schema while preserving complete schema and relationships."""
    active_db = st.session_state.get("active_db")
    if active_db is None:
        active_db = get_default_database()

    active_retriever = st.session_state.get("active_retriever")
    if active_retriever is None:
        active_retriever = build_retriever(
            active_db,
            st.session_state.get("relationship_context", "")
        )
        st.session_state.active_retriever = active_retriever

    docs = active_retriever.invoke(question)
    retrieved_context = "\n\n".join(doc.page_content for doc in docs)
    relationship_context = st.session_state.get("relationship_context", "")

    try:
        tables = active_db.get_usable_table_names()
        full_schema = active_db.get_table_info(tables)
        return (
            "COMPLETE ACTIVE DATABASE SCHEMA:\n"
            + full_schema
            + "\n\n"
            + relationship_context
            + "\n\nRELEVANT RAG CONTEXT:\n"
            + retrieved_context
        )
    except Exception:
        return relationship_context + "\n\n" + retrieved_context



def get_conversation_context():

    conversation_context = st.session_state.get(
        "conversation_context",
        []
    )

    if not conversation_context:
        return "No previous conversation."

    context = ""

    for item in conversation_context[-5:]:

        context += (
            f"Previous User Question: {item['question']}\n"
            f"Generated SQL: {item['sql']}\n"
            f"Previous Answer: {item['answer']}\n\n"
)

    return context


# SQL GENERATION PROMPT

sql_prompt = PromptTemplate.from_template(
    """
You are an expert data analyst and SQL developer.
Convert the user's natural-language question into ONE valid, read-only SELECT query.

Active Database Dialect:
{dialect}

Relevant Database Schema and Sample Data:
{context}

Conversation Context:
{conversation}

User Question:
{question}

Rules:
1. Return only one SQL query. No explanation.
2. Use only tables and columns present in the COMPLETE ACTIVE DATABASE SCHEMA or RAG context.
3. Use AUTOMATICALLY DETECTED TABLE RELATIONSHIPS as guidance when a multi-table JOIN is needed. Treat them as likely relationships, not guaranteed constraints.
4. Never invent tables, columns, relationships, or values.
5. Use valid syntax for the active database dialect.
6. Generate read-only SELECT statements only.
7. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT or REVOKE.
8. Use COUNT, SUM, AVG, MIN and MAX when appropriate.
9. Use GROUP BY for category comparisons and ORDER BY/LIMIT for rankings.
10. Use WHERE filters with values that actually occur in the relevant column.
11. Resolve follow-up references such as it, its, this, that, they and their from conversation context.
12. NEVER use SELECT NULL as a substitute for answering the question. If a relevant column exists, query that column.
13. If the question asks for a numeric aggregate, identify the numeric column(s) from the schema and calculate the requested aggregate.
14. If the question asks for an average of numeric columns, return one row containing AVG for each applicable numeric column, with clear aliases.
15. If the question asks for the most common/top value of a categorical column, use COUNT(*), GROUP BY that column, ORDER BY COUNT(*) DESC and LIMIT.
16. If the question asks for the top records by a numeric field, identify the numeric field from the schema and use ORDER BY that field DESC LIMIT.
17. If the question names a job, product, person, category, location, etc., match it against the appropriate text column based on the schema and sample values.
18. If a text column contains delimited multiple values, use a dialect-supported string-splitting approach when individual values must be counted.
19. Prefer simple executable SQL over clever or unnecessary SQL.

SQL Query:
"""
)

# SQL CHAIN

sql_chain = (
    {
        "context": lambda x: get_rag_context(x["question"]),
        "conversation": lambda x: get_conversation_context(),
        "question": lambda x: x["question"],
        "dialect": lambda x: get_default_database().dialect
    }
    | sql_prompt
    | llm
    | StrOutputParser()
)

# CLEAN GENERATED SQL


def clean_sql(sql_response):
    sql = str(sql_response).strip()
    sql = re.sub(r"```(?:sql|sqlite|mysql)?", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "").strip()
    match = re.search(r"(?is)\b(?:WITH|SELECT)\b.*", sql)
    if not match:
        return ""
    sql = match.group(0).strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return sql.strip()

# VALIDATE SQL

def apply_numeric_rounding(query):
    """Round SUM-based numeric aggregates to two decimal places."""
    if re.search(r"(?i)\bROUND\s*\(\s*SUM\s*\(", query):
        return query

    return re.sub(
        r"(?i)\bSUM\s*\(\s*((?:[A-Za-z_][A-Za-z0-9_]*\.)?`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*)\s*\)",
        lambda m: f"ROUND(SUM({m.group(1)}), 2)",
        query
    )


def validate_sql(query):
    """Allow read-only SELECT/CTE queries and reject database mutations."""
    if not query or not str(query).strip():
        return False

    query_text = str(query).strip()

    # Remove leading SQL comments before checking the statement type.
    query_text = re.sub(r"(?is)^(?:\s*--[^\n]*\n|\s*/\*.*?\*/\s*)+", "", query_text).strip()

    # A WITH query is also read-only when its final statement is SELECT.
    starts_read_only = bool(re.match(r"(?is)^(SELECT|WITH)\b", query_text))
    if not starts_read_only:
        return False

    # Ignore quoted strings and SQL comments while checking forbidden keywords.
    # This prevents harmless values such as 'CREATE' from being treated as
    # database modification commands.
    check_text = re.sub(r"(?is)'(?:''|[^'])*'|\"(?:\"|[^\"])*\"", " ", query_text)
    check_text = re.sub(r"(?m)--[^\n]*", " ", check_text)
    check_text = re.sub(r"(?is)/\*.*?\*/", " ", check_text)

    forbidden_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "TRUNCATE", "CREATE", "GRANT", "REVOKE",
        "REPLACE", "MERGE", "CALL", "EXEC", "EXECUTE"
    ]

    return not any(
        re.search(rf"\b{keyword}\b", check_text, flags=re.IGNORECASE)
        for keyword in forbidden_keywords
    )

# corrected sql 

def correct_sql(question, query, error_message):

    correction_prompt = PromptTemplate.from_template(
        """
You are an expert SQL developer.

The following SQL query failed during database execution.

Active Database Dialect:
{dialect}

User Question:
{question}

Generated SQL:
{query}

Database Error:
{error}

Database Schema:
{context}

Generate a corrected SELECT query for the active database dialect.

Rules:
1. Return only the corrected SQL query.
2. Use only tables and columns present in the provided schema.
3. Do not invent table names or column names.
4. Use valid SQL syntax for the active database dialect.
5. Use SELECT queries only.
6. Do not modify the database.
7. Do not use INSERT, UPDATE, DELETE, DROP, ALTER,
   TRUNCATE, CREATE, GRANT or REVOKE.
8. Correct the query based on the database error.

Corrected SQL:
"""
    )

    correction_chain = (
        correction_prompt
        | llm
        | StrOutputParser()
    )

    corrected_sql = correction_chain.invoke(
        {
            "question": question,
            "query": query,
            "error": error_message,
            "context": get_rag_context(question),
            "dialect": get_default_database().dialect
        }
    )

    return clean_sql(corrected_sql)

# ANSWER GENERATION PROMPT

answer_prompt = PromptTemplate.from_template(
    """
You are a helpful data analyst.

Answer the user's question using ONLY the database result
provided below.

User Question:
{question}

SQL Query:
{query}

Database Result:
{result}

Rules:

1. Answer the user's question directly.
2. Use only information contained in the database result.
3. Do not make up information.
4. Keep the answer clear and concise.
5. Do not explain the SQL unless the user asks.
6. Do not assume currency, units, dates, or other details
   that are not present in the query or result.
7. If the database result is empty, clearly state that
   no matching record was found.
8. Format numeric values with commas and round long decimal
   values to two decimal places when appropriate.
9. Keep the final answer natural and concise.

Final Answer:
"""
)


# ANSWER CHAIN


answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


def is_modification_request(question):

    modification_keywords = [
        "insert",
        "add",
        "update",
        "delete",
        "remove",
        "drop",
        "alter",
        "truncate",
        "create"
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in modification_keywords
    )


# MAIN CHATBOT FUNCTION


def resolve_followup_question(question):
    """Resolve simple conversational references before sending the question to the LLM."""

    context = st.session_state.get("conversation_context", [])

    if not context:
        return question

    question_lower = question.lower()

    reference_words = [
        " it ", " its ", " it's ", " that ", " this ",
        " they ", " their "
    ]

    has_reference = any(word in f" {question_lower} " for word in reference_words)

    if not has_reference:
        return question

    latest = context[-1]
    previous_answer = str(latest.get("answer", "")).strip()

    # Remove common punctuation from a short entity-style answer.
    entity = previous_answer.strip(" .!?\"'")

    # Use only short, clear answers as references. This avoids replacing
    # pronouns with an entire explanatory sentence.
    if not entity or len(entity) > 80 or "\n" in entity:
        return question

    # If the previous answer is a simple entity such as "Wholesale",
    # rewrite the follow-up into an explicit question.
    resolved = re.sub(
        r"\bits\b|\bit's\b|\bit\b",
        f"for {entity}",
        question,
        flags=re.IGNORECASE
    )

    return resolved


def format_answer_numbers(text):
    def replace_number(match):
        try:
            return f"{float(match.group(0).replace(',', '')):,.2f}"
        except ValueError:
            return match.group(0)
    pattern = r"(?<![A-Za-z0-9_])-?\d{1,3}(?:,\d{3})*\.\d{3,}|(?<![A-Za-z0-9_])-?\d+\.\d{3,}"
    return re.sub(pattern, replace_number, str(text))


def text_to_sql_chatbot(question):


    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []


    if is_modification_request(question):
        return {
            "question": question,
            "sql": None,
            "result": None,
            "answer": (
                "I can only execute read-only database queries. "
                "INSERT, UPDATE, DELETE and other database "
                "modification operations are not supported."
            )
        }

    # Resolve simple follow-up references such as
    # "What is its total sales?" using the previous answer.
    original_question = question
    question = resolve_followup_question(question)

    # Generate SQL
    sql_response = sql_chain.invoke(
        {
            "question": question
        }
    )

    # Clean SQL
    query = clean_sql(sql_response)
    query = apply_numeric_rounding(query)

    # Validate SQL. If the first generation is not safely read-only, give the
    # correction model one chance to regenerate it instead of failing immediately.
    if not validate_sql(query):
        try:
            corrected_query = correct_sql(
                question,
                query or "(no usable SQL was generated)",
                "The generated statement was not a valid read-only SELECT/CTE query."
            )
            corrected_query = apply_numeric_rounding(corrected_query)

            if validate_sql(corrected_query):
                query = corrected_query
            else:
                return {
                    "question": question,
                    "sql": query,
                    "result": None,
                    "answer": "I could not generate a safe SQL query."
                }
        except Exception:
            return {
                "question": question,
                "sql": query,
                "result": None,
                "answer": "I could not generate a safe SQL query."
            }

    # Execute SQL

    try:

        result = get_default_database().run(query)

    except Exception as e:

        # Try to correct the SQL query
        try:

            corrected_query = correct_sql(
                question,
                query,
                str(e)
            )
            corrected_query = apply_numeric_rounding(corrected_query)

            # Validate corrected SQL
            if not validate_sql(corrected_query):

                return {
                    "question": question,
                    "sql": query,
                    "result": None,
                    "answer": (
                        "The generated SQL query could not be "
                        "corrected safely."
                    )
                }

            # Execute corrected SQL
            result = get_default_database().run(corrected_query)

            # Update query shown to user
            query = corrected_query

        except Exception as correction_error:

            return {
                "question": question,
                "sql": query,
                "result": None,
                "answer": (
                    "Sorry, I could not execute or "
                    "correct the SQL query.\n\n"
                    f"Error: {correction_error}"
                )
            }
    # Generate final answer
    try:

        answer = answer_chain.invoke(
            {
                "question": question,
                "query": query,
                "result": result
            }
        )

    except Exception as e:

        return {
            "question": question,
            "sql": query,
            "result": result,
            "answer": (
                "SQL executed successfully, but I could not "
                "generate the final answer.\n\n"
                f"Error: {e}"
            )
        }

    answer = format_answer_numbers(answer)

    st.session_state.conversation_context.append(
    {
        "question": question,
          "sql": query,
        "answer": answer

    }
)
    
    return {
        "question": original_question,
        "sql": query,
        "result": result,
        "answer": answer
    }


# SIDEBAR


with st.sidebar:

    st.markdown(
        """<div class='sidebar-brand'>
            <div class='brand-row'>
                <div class='brand-mark'>✦</div>
                <div>
                    <div class='brand-name'>DataPilot AI</div>
                    <div class='brand-sub'>Talk to your data, naturally</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("<div class='sidebar-section'>Choose your data</div>", unsafe_allow_html=True)

    try_demo = st.button(
        "✦  Try Demo Dataset",
        key="try_demo",
        use_container_width=True,
        help="Load the bundled demo dataset and start querying immediately."
    )

    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload 1–5 related CSV files. Each file becomes a separate SQLite table."
    )

    if len(uploaded_files) > 5:
        st.error("Please upload at most 5 CSV files at a time.")
        uploaded_files = uploaded_files[:5]

    if try_demo:
        try:
            with st.spinner("Loading demo tables and building relationship-aware RAG index..."):
                st.session_state.active_db = None
                st.session_state.active_retriever = None
                st.session_state.csv_hash = None
                st.session_state.demo_hash = None
                st.session_state.relationship_context = ""
                st.session_state.relationship_count = 0
                st.session_state.uploaded_files = []
                st.session_state.data_mode = "demo"
                get_demo_database()
                st.session_state.messages = []
                st.session_state.conversation_context = []
        except Exception as e:
            st.error(f"Could not load demo dataset: {e}")

    elif uploaded_files:
        file_items = [
            (uploaded_file.name, uploaded_file.getvalue())
            for uploaded_file in uploaded_files
        ]
        combined_hash = hashlib.md5()
        for filename, csv_bytes in file_items:
            combined_hash.update(filename.encode("utf-8"))
            combined_hash.update(csv_bytes)
        csv_hash = combined_hash.hexdigest()

        if st.session_state.csv_hash != csv_hash or st.session_state.data_mode != "csv":
            try:
                with st.spinner("Loading tables, detecting relationships and building RAG index..."):
                    csv_db, relationship_context, relationships = build_multi_csv_database(file_items)
                    csv_retriever = build_retriever(csv_db, relationship_context)

                st.session_state.active_db = csv_db
                st.session_state.active_retriever = csv_retriever
                st.session_state.active_source = "Custom CSV Dataset"
                st.session_state.csv_hash = csv_hash
                st.session_state.demo_hash = None
                st.session_state.data_mode = "csv"
                st.session_state.relationship_context = relationship_context
                st.session_state.relationship_count = len(relationships)
                st.session_state.uploaded_files = [name for name, _ in file_items]
                st.session_state.messages = []
                st.session_state.conversation_context = []
            except Exception as e:
                st.error(f"Could not load CSV files: {e}")
        else:
            st.session_state.uploaded_files = [name for name, _ in file_items]

    elif st.session_state.get("data_mode") != "demo":
        # If the user removes their custom CSV dataset, return to the bundled demo.
        st.session_state.active_db = None
        st.session_state.active_retriever = None
        st.session_state.active_source = "Demo Dataset"
        st.session_state.csv_hash = None
        st.session_state.relationship_context = ""
        st.session_state.relationship_count = 0
        st.session_state.uploaded_files = []
        st.session_state.data_mode = "demo"

    # Make the bundled multi-table demo the public default when nothing is selected.
    if st.session_state.get("data_mode") == "demo" and st.session_state.get("active_db") is None:
        try:
            with st.spinner("Preparing demo tables and relationship-aware RAG..."):
                get_demo_database()
        except Exception as e:
            st.error(f"Could not load demo dataset: {e}")

    source_label = st.session_state.active_source
    if st.session_state.get("data_mode") == "csv":
        source_type = "Your CSVs · SQLite"
    elif st.session_state.get("data_mode") == "demo":
        source_type = "Demo Dataset · SQLite"
    else:
        source_type = "MySQL"

    st.markdown(
        f"<div class='source-card'><div class='source-label'>ACTIVE DATA SOURCE</div>"
        f"<div class='source-name'>{source_type}</div><div class='source-detail'>{source_label}</div></div>",
        unsafe_allow_html=True
    )

    if st.session_state.get("data_mode") in {"demo", "csv"}:
        try:
            active_tables = (st.session_state.get("active_db") or get_default_database()).get_usable_table_names()
            st.caption("Tables: " + ", ".join(active_tables))
            st.caption(
                f"{len(active_tables)} table(s) · "
                f"{st.session_state.get('relationship_count', 0)} relationship(s) detected"
            )
        except Exception:
            pass

    st.divider()
    st.markdown("<div class='section-kicker'>System</div><div class='section-title'>Status</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='active-source'><span class='active-dot'></span><span class='active-title'>System online</span>"
        "<div class='active-detail'>RAG + FAISS · LangChain · Groq · read-only SQL</div></div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("<div class='sidebar-section'>Quick start</div>", unsafe_allow_html=True)

    if st.session_state.get("data_mode") == "demo":
        examples = [
            "How many records are there in each table?",
            "Which table contains the most records?",
            "What tables are available?",
            "Show the relationship between the tables."
        ]
    else:
        examples = [
            "How many records are there in each table?",
            "What tables are available?",
            "Which tables appear to be related?",
            "Show the top 5 records from the main table."
        ]

    for i, example in enumerate(examples):
        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):
            st.session_state.pending_question = example

    st.divider()
    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

# MAIN HEADER

st.markdown(
    """<div class='hero'>
        <div class='hero-content'>
            <div class='hero-eyebrow'>AI-POWERED DATA QUERYING</div>
            <div class='hero-title'>Talk to your data.</div>
            <div class='hero-text'>Ask questions in plain English. The assistant finds the relevant schema, generates a read-only SQL query, runs it on your active data source, and turns the result into a clear answer.</div>
            <div class='hero-pill'>✦ RAG · FAISS · LangChain · Groq · MySQL / CSV</div>
        </div>
    </div>""",
    unsafe_allow_html=True
)

active_display = ("Your CSVs / SQLite" if st.session_state.get("data_mode") == "csv" else "Demo Dataset / SQLite" if st.session_state.get("data_mode") == "demo" else "MySQL")

# SYSTEM OVERVIEW

card1, card2, card3, card4 = st.columns(4)

active_display = ("Your CSVs / SQLite" if st.session_state.get("data_mode") == "csv" else "Demo Dataset / SQLite" if st.session_state.get("data_mode") == "demo" else "MySQL")

with card1:
    st.markdown("<div class='info-card'><div class='info-label'>Language model</div><div class='info-value'>GPT-OSS-20B</div><div class='info-note'>Groq API</div></div>", unsafe_allow_html=True)
with card2:
    st.markdown("<div class='info-card'><div class='info-label'>Retrieval</div><div class='info-value'>RAG + FAISS</div><div class='info-note'>Schema-aware retrieval</div></div>", unsafe_allow_html=True)
with card3:
    st.markdown("<div class='info-card'><div class='info-label'>Workflow</div><div class='info-value'>LangChain</div><div class='info-note'>Text-to-SQL pipeline</div></div>", unsafe_allow_html=True)
with card4:
    st.markdown(f"<div class='info-card'><div class='info-label'>Active source</div><div class='info-value'>{active_display}</div><div class='info-note'>Ready for queries</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# SUGGESTED QUESTIONS


st.markdown("<div class='section-kicker'>Explore</div><div class='section-title'>Try a query</div><div class='section-subtitle'>Start with an example or ask anything about the active data source.</div>", unsafe_allow_html=True)

query_cols = st.columns(4)

if st.session_state.active_source != "MySQL":
    suggestions = [
        "How many records are there in each table?",
        "What tables are available?",
        "Which tables appear to be related?",
        "Show the top 5 records from a table."
    ]
else:
    suggestions = [
        "What is the total sales amount?",
        "How many orders are there?",
        "Which sales channel has the highest sales?",
        "What was the budget of Product 12?"
    ]


for i, suggestion in enumerate(suggestions):

    with query_cols[i]:

        if st.button(
            suggestion,
            key=f"suggestion_{i}",
            use_container_width=True
        ):

            st.session_state.pending_question = suggestion


# CHAT HISTORY


for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )

            if message.get("sql"):

                with st.expander(
                    "View Generated SQL"
                ):

                    st.code(
                        message["sql"],
                        language="sql"
                    )

            if message.get("result") is not None:

                with st.expander(
                    "View Database Result"
                ):

                    st.write(
                        message["result"]
                    )

# CHAT INPUT


user_question = st.chat_input(
    "Ask a question about your database..."
)

# HANDLE EXAMPLE QUESTION


if (
    st.session_state.pending_question
    and not user_question
):

    user_question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None

# PROCESS USER QUESTION


if user_question:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Process question
    with st.spinner(
        "Retrieving schema and generating answer..."
    ):

        response = text_to_sql_chatbot(
            user_question
        )

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sql": response["sql"],
            "result": response["result"]
        }
    )

    st.rerun()

# HOW IT WORKS

st.markdown("<div class='section-kicker'>Under the hood</div><div class='section-title'>How the system works</div><div class='section-subtitle'>A simple read-only pipeline from natural language to database answer.</div>", unsafe_allow_html=True)

steps = [
    ("01", "Question", "You ask in plain English"),
    ("02", "RAG", "Relevant schema is retrieved"),
    ("03", "LLM", "SQL is generated"),
    ("04", "Validation", "Unsafe SQL is blocked"),
    ("05", "Database", "Query runs on active source"),
    ("06", "Answer", "Result is explained"),
]

step_cols = st.columns(6)
for col, (num, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""<div class='info-card' style='min-height:112px'>
                <div class='info-label'>{num}</div>
                <div class='info-value'>{title}</div>
                <div class='info-note'>{desc}</div>
            </div>""",
            unsafe_allow_html=True
        )


# FOOTER


st.divider()

st.markdown(
    "<div class='footer'>Text-to-SQL AI Assistant · RAG + FAISS + LangChain + Groq · MySQL / CSV · Read-only querying</div>",
    unsafe_allow_html=True
)