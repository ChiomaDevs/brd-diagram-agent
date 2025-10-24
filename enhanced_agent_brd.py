import os
import sys
from dotenv import load_dotenv
import json  # NEW: For cleaner JSON handling

# --------- Diagnostics ----------
def _print_versions():
    try:
        import langchain as lc
        import langchain_openai as lco
        print(f"[info] python={sys.version.split()[0]}",
              f"langchain={getattr(lc, '__version__', 'unknown')}",
              f"langchain-openai={getattr(lco, '__version__', 'unknown')}")
    except Exception as e:
        print("[info] version check failed:", e)
_print_versions()

# --------- Imports ----------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Tool import
try:
    from langchain_core.tools import Tool
except Exception:
    from langchain.tools import Tool

# NEW: For PDF reading and Mermaid rendering
try:
    import pypdf2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[warn] Install PyPDF2 for PDF support: pip install pypdf2")

try:
    from mermaid import Mermaid  # From Step 2's mermaid-py
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False
    print("[warn] Install mermaid-py for auto-rendering: pip install mermaid-py")

# --------- API key & Model ----------
# UPDATED: Lazy load API key for Streamlit Cloud (from st.secrets or .env fallback)
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")  # Local fallback
    if not api_key and 'streamlit' in sys.modules:  # Cloud check
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError("❌ OpenAI API key missing! Add to .env (local) or secrets.toml (cloud).")
    return api_key

# Model (now uses lazy key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=get_api_key())  # Calls on init

# --------- Prompts & Chains (Unchanged from Your Code) ----------
parse_prompt = PromptTemplate(
    input_variables=["brd_text"],
    template="""
You are a requirements analyst. Extract key elements from this BRD text for diagrams:
- PROCESSES: Main steps or actions (e.g., "Submit Issue").
- DATA_FLOWS: How data moves (e.g., "User -> System").
- RULES: Business logic/decisions (e.g., "If external, allocate to admin").
- ENTITIES: Data objects/tables (e.g., "User", "Issue").

BRD: {brd_text}

Respond in JSON only, like:
{{
  "processes": ["Process1", "Process2"],
  "data_flows": ["Flow1: A -> B", "Flow2: C -> D"],
  "rules": ["Rule1: If X then Y", "Rule2: Else Z"],
  "entities": ["Entity1", "Entity2"]
}}
"""
)
parse_chain = parse_prompt | llm

dfd_prompt = PromptTemplate(
    input_variables=["extracted"],
    template="""
Create a simple Data Flow Diagram (DFD) in Mermaid from this extracted info:
{extracted}

Use flowchart syntax like:
flowchart TD
  A[User] --> B[Submit]
  B --> C[System]

Keep it to 5-10 nodes max. Output ONLY the Mermaid code.
"""
)
dfd_chain = dfd_prompt | llm

logic_prompt = PromptTemplate(
    input_variables=["extracted"],
    template="""
Create a decision flowchart in Mermaid from the RULES within:
{extracted}

Use:
graph TD
  Start([Start]) --> IsExternal{{External?}}
  IsExternal -->|Yes| Allocate[Allocate to Admin]
  IsExternal -->|No| Queue[Queue for Triage]

Output ONLY the Mermaid code.
"""
)
logic_chain = logic_prompt | llm

db_prompt = PromptTemplate(
    input_variables=["extracted"],
    template="""
Create an Entity-Relationship Diagram (ERD) in Mermaid from the entities and relationships implied here:
{extracted}

Use: erDiagram; with tables, PK/FK, and relationships like ||--o{{ .
Output ONLY the Mermaid code, e.g.:
erDiagram
  USER {{
    int id PK
    string name
  }}
  ISSUE {{
    int id PK
    int user_id FK
    string title
  }}
  USER ||--o{{ ISSUE : "raises"
"""
)
db_chain = db_prompt | llm

# --------- Tool functions (Unchanged) ----------
def parse_brd_tool(input_text: str) -> str:
    msg = parse_chain.invoke({"brd_text": input_text})
    return getattr(msg, "content", str(msg))

def generate_dfd_tool(extracted_json: str) -> str:
    return dfd_chain.invoke({"extracted": extracted_json}).content

def generate_logic_tool(extracted_json: str) -> str:
    return logic_chain.invoke({"extracted": extracted_json}).content

def generate_db_tool(extracted_json: str) -> str:
    return db_chain.invoke({"extracted": extracted_json}).content

tools = [
    Tool(name="ParseBRD", description="Extract processes, flows, rules, entities from BRD text. Input: BRD text string.", func=parse_brd_tool),
    Tool(name="GenDFD", description="Create DFD Mermaid code. Input: Extracted JSON from ParseBRD.", func=generate_dfd_tool),
    Tool(name="GenLogic", description="Create Logic Flowchart Mermaid code. Input: Extracted JSON from ParseBRD.", func=generate_logic_tool),
    Tool(name="GenDB", description="Create DB ERD Mermaid code. Input: Extracted JSON from ParseBRD.", func=generate_db_tool),
]

# --------- Build Agent (Unchanged) ----------
def build_agent(llm, tools):
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert BRD Diagram Agent. Take a BRD text, parse it, and generate Mermaid diagrams (DFD, Logic, DB).
Think step-by-step. Use tools when needed: ParseBRD, GenDFD, GenLogic, GenDB.
"""),
            ("human", "User input: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    except Exception:
        pass

    try:
        from langchain.agents import initialize_agent, AgentType
        system_text = """
You are an expert BRD Diagram Agent. Take a BRD (text/file), parse it, and generate Mermaid diagrams (DFD, Logic, DB).
1) Use ParseBRD to extract JSON; 2) call Gen* tools as needed; 3) ask minimal clarifying questions if key info is missing.
"""
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": system_text,
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="agent_scratchpad")],
            },
        )
    except Exception:
        return None

agent = build_agent(llm, tools)

# --------- Enhanced Helpers ----------
def read_file_if_needed(user_input: str) -> str:
    """NEW: Now handles PDFs too!"""
    if user_input.endswith(('.txt', '.md')) and os.path.exists(user_input):
        with open(user_input, 'r', encoding='utf-8') as f:
            return f.read()
    elif PDF_AVAILABLE and user_input.endswith('.pdf') and os.path.exists(user_input):
        try:
            with open(user_input, 'rb') as f:
                reader = pypdf2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"[warn] PDF read failed: {e}. Treating as text.")
            return user_input
    return user_input

def render_and_save_mermaid(mermaid_code: str, diagram_type: str, output_dir: str = "output"):
    """NEW: Render Mermaid code to SVG/PNG and save."""
    if not MERMAID_AVAILABLE:
        return mermaid_code  # Fallback to text

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{diagram_type.lower()}.svg"
    file_path = os.path.join(output_dir, file_name)

    try:
        diagram = Mermaid(mermaid_code)
        diagram.svg(outputfile=file_path)
        print(f"💾 Saved {diagram_type} image: {file_path}")
        return f"Rendered image saved: {file_path}\nCode: {mermaid_code}"
    except Exception as e:
        print(f"[warn] Rendering failed: {e}")
        return mermaid_code

# --------- Run (Enhanced Output) ----------
if __name__ == "__main__":
    try:
        user_input = input("Enter BRD text, or file path (e.g., my_brd.pdf or .txt): ")
    except EOFError:
        user_input = "Sample BRD: Users submit issues; admins triage; system assigns owners; statuses update; reports generated."
    
    brd_text = read_file_if_needed(user_input)
    if not brd_text.strip():
        print("No input—try again!")
        sys.exit(0)

    print("🤖 Agent starting...")
    if agent is not None:
        result = agent.invoke({"input": brd_text})
        output = result if isinstance(result, str) else result.get("output", str(result))
    else:
        # Fallback: Direct pipeline + NEW rendering/saving
        print("[info] Agent API not available. Running direct pipeline…")
        extracted = parse_brd_tool(brd_text)
        print("\n[Extracted JSON]\n", extracted)
        
        dfd = generate_dfd_tool(extracted)
        print("\n[DFD]\n", render_and_save_mermaid(dfd, "DFD"))
        
        logic = generate_logic_tool(extracted)
        print("\n[Logic]\n", render_and_save_mermaid(logic, "Logic"))
        
        db = generate_db_tool(extracted)
        print("\n[ERD]\n", render_and_save_mermaid(db, "ERD"))
        
        output = "Pipeline complete with renders!"


    print(f"\n✨ Final Output:\n{output}")