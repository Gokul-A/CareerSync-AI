import streamlit as st
import os
import shutil
import sqlite3
import json
import tempfile
import operator
from datetime import datetime
from typing import Annotated, List, TypedDict, Dict, Any

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter
from sys_env import CSI_ID, google_search_key, groq_key, open_ai_key


DB_FOLDER = "JOB_DB_NEW/"
os.makedirs(os.path.join(DB_FOLDER, 'structured'), exist_ok=True)
os.makedirs(os.path.join(DB_FOLDER, 'vectors'), exist_ok=True)
DB_PATH = f"{DB_FOLDER}/structured/jobs.sqlite"
VECTOR_DB_PATH = f"{DB_FOLDER}/vectors/jobs_chroma"


_MODEL = None
def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL
    
def get_job_vectorstore():
    return Chroma(
        collection_name="jobs",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=None  # we embed manually
    )


def init_db(db_path):
    """Initializes the jobs table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with 'link' as the PRIMARY KEY to prevent duplicates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            link TEXT PRIMARY KEY,
            role TEXT,
            company TEXT,
            job_posted_date TEXT,
            skills TEXT,
            summary TEXT,
            found_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db(DB_PATH)

class JobPosting(BaseModel):
    """Schema for a single job posting extracted from search results."""
    role: str = Field(description="The job title or role (e.g., 'Data Scientist')")
    company: str = Field(description="The name of the hiring company")
    experience: str = Field(description="Minimum required experience")
    skills: List[str] = Field(description="List of required technical skills (e.g., Python, SQL, NLP)")
    link: str = Field(description="The direct URL to the job posting")
    JobPostedDate: str = Field(description="Date of job posting (YYYY-MM-DD)")
    summary: str = Field(description="A comprehensive summary of the job description in 10-12 sentences")

# If you expect multiple jobs from one search, define a wrapper
class JobSearchResponse(BaseModel):
    """
    Structure for the final output containing a list of extracted job postings.
    """
    jobs: List[JobPosting] = Field(description="A list of job postings found")


class JobCompatibility(BaseModel):
    """
    Structure for the output containing a list of extracted job postings and compatibility score based on user profile.
    """
    role: str = Field(description="The job role title")
    company: str = Field(description="The company name")
    compatibility_score: int = Field(description="A score from 0-100 indicating how well the resume matches the job description")
    missing_skills: List[str] = Field(description="List of key skills found in the job description but missing from the resume")
    reasoning: str = Field(description="Why this job is suitable for the user (under 200 chars)")
    recruiter_message: str = Field(description="A concise, professional, and customized message (under 500 chars) from the candidate with introduction to send to the recruiter to describe why candidate is fit for this job.")
    apply_link: str = Field(description="The original URL to apply for the job")

# This helps us wrap the final list for the state
class CompatibilityReport(BaseModel):
    """
    Structure for the final output containing a list of compatible job postings.
    """
    matches: List[JobCompatibility]


# --- 1. UNIFIED STATE DEFINITION ---
class UnifiedState(TypedDict):
    # Mode selection
    mode: str  # "search", "match", "view"
    
    # Search Data
    query: str
    db_path: str
    job_links: List[str]
    scraped_content: List[str]
    extracted_jobs: Annotated[List[JobPosting], operator.add]
    new_jobs_count: int
    
    db_jobs: List[Dict[str, Any]]
    
    # Match Data
    resume_path: str
    resume_text: str
    resume_embedding: List[float]
    job_ids: List[str]
    raw_jobs: List[dict]
    analysis_results: List[JobCompatibility]

# --- NODES: SEARCH & EXTRACT ---

g_search = GoogleSearchAPIWrapper(
    google_api_key=google_search_key,
    google_cse_id=CSI_ID,
    k=3)

def search_node(state: UnifiedState):
    
    ''' Search Google, filter out previously seen jobs, and return fresh links.'''
    
    query = state["query"]
    
    
    # Simple duplicate check against DB
    conn = sqlite3.connect(state['db_path'])
    cursor = conn.cursor()
    cursor.execute("SELECT link FROM jobs")
    seen_links = {row[0] for row in cursor.fetchall()}
    conn.close()

    print(f"--- Memory Check: {len(seen_links)} jobs already seen ---")

    results = g_search.results(query, num_results=3)
    if not results:
        return {"job_links": []}

    # 3. Filter out links that are in seen_links
    fresh_links = []
    skipped_count = 0

    for r in results:
        url = r.get("link")
        if url and url not in seen_links:
            fresh_links.append(url)
        else:
            skipped_count += 1

    print(f"--- Skipped {skipped_count} duplicates. Found {len(fresh_links)} fresh links. ---")


    if len(fresh_links) == 0:
        print("--- All top results were duplicates. Attempting deeper search... ---")

        results_deep = g_search.results(query, num_results=15)

        # Process the new batch (checking only the newly fetched ones)
        for r in results_deep:
            url = r.get("link")
          # We check if it's NOT in seen_links AND NOT already in fresh_links (from previous loop)
            if url and url not in seen_links and url not in fresh_links:
                fresh_links.append(url)

    # 5. Return only the requested number (k=3)
    final_links = fresh_links[:3]

    print(f"--- Returning top {len(final_links)} fresh links ---")
    return {"job_links": final_links}



def scrape_node(state: UnifiedState):
    """Visits the links and extracts raw text."""
    
    links = state["job_links"]
    scraped_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for link in links:
        try:
            loader = WebBaseLoader([link], header_template=headers)
            docs = loader.load()
            for doc in docs:
                content = f"SOURCE URL: {doc.metadata['source']}\n\nCONTENT:\n{doc.page_content[:3000]}"
                scraped_data.append(content)
        except Exception:
            pass
    return {"scraped_content": scraped_data}

def extraction_node(state: UnifiedState):
    """Extract structured data from scraped text."""
    
    scraped_content = state["scraped_content"]
    # Using OpenAI or Groq here
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_ai_key, temperature=0)
    structured_llm = llm.with_structured_output(JobSearchResponse)
    
    all_jobs = []
    for page_text in scraped_content:
        messages = [
            SystemMessage(content="""
                You are an expert extraction algorithm.
                Extract the job details from the text.
                Extract only the relevant job from the user's input
                IMPORTANT: Do NOT wrap the output in markdown code blocks (like ```json).
                Output RAW JSON only."""),
            HumanMessage(content="User Query: \n"+state["query"]),
            HumanMessage(content=page_text),
        ]
        try:
            result = structured_llm.invoke(messages)
            if result and result.jobs:
                all_jobs.extend(result.jobs)
        except Exception as e:
            print(f"Extraction error: {e}")
            continue
            
    return {"extracted_jobs": all_jobs}

def save_new_jobs(state: UnifiedState):
    """
    Saves jobs to the DB. Skips duplicates automatically due to PRIMARY KEY.
    Returns the count of new jobs added.
    """
    
    conn = sqlite3.connect(state['db_path'])
    cursor = conn.cursor()
    count = 0
    for job in state['extracted_jobs']:
        try:
            skills_str = json.dumps(job.skills)
            cursor.execute('''
                INSERT INTO jobs (link, role, company, job_posted_date, skills, summary, found_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (job.link, job.role, job.company, job.JobPostedDate, skills_str, job.summary, datetime.now().strftime("%Y-%m-%d")))
            count += 1
        except sqlite3.IntegrityError:
            print(f"Skipping duplicate: {job.role} at {job.company}")
            pass
    conn.commit()
    conn.close()
    return {'new_jobs_count': count}

def embed_and_store_jobs(state: UnifiedState):
    
    """Saves jobs to the Vector DB. Skips duplicates."""
    
    model = get_embedding_model()
    vectordb = get_job_vectorstore()
    
    seen_ids = set()
    unique_jobs = []
    
    for job in state["extracted_jobs"]:
        if job.link not in seen_ids:
            unique_jobs.append(job)
            seen_ids.add(job.link)
        else:
            print(f"Skipping duplicate: {job.role} at {job.company}")
        
    texts = []
    metadatas = []
    ids = []

    for job in unique_jobs:
        text = f"""
                    Role: {job.role}
                    Company: {job.company}
                    Experience: {job.experience}
                    Skills: {', '.join(job.skills)}
                    Summary: {job.summary}
                """
        
        texts.append(text)
        metadatas.append({"job_id": job.link, "role": job.role, "company": job.company})
        ids.append(job.link)

    embeddings = model.encode(texts, normalize_embeddings=True)

    vectordb._collection.upsert(
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts,
        ids=ids
    )

    return state

# --- NODES: VIEW ---
def view_jobs_node(state: UnifiedState):
    """Fetch all jobs from the SQLite database."""
    conn = sqlite3.connect(state["db_path"])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY found_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return {"db_jobs": [dict(row) for row in rows]}

# --- NODES: MATCHING ---
def read_resume_pdf(file_path):
    """Extract text from the uploaded PDF resume."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_text()

def embed_resume(state: UnifiedState):
    """Converts the raw resume text to vector embedding."""
    state["resume_text"] = read_resume_pdf(state['resume_path'])
    model = get_embedding_model()
    state["resume_embedding"] = model.encode(state["resume_text"], normalize_embeddings=True).tolist()
    return state

def retrieve_similar_jobs(state: UnifiedState):
    """Retrive matching jobs from vector DB."""
    vectordb = get_job_vectorstore()
    if not state["resume_embedding"]: return {"job_ids": []}
    results = vectordb.similarity_search_by_vector(state["resume_embedding"], k=5)
    state["job_ids"] = [doc.metadata["job_id"] for doc in results]
    return state

def load_jobs_for_matching(state: UnifiedState):
    """Retrive all matching jobs from sql db"""
    if not state["job_ids"]: 
        return {"raw_jobs": []}
    conn = sqlite3.connect(state['db_path'])
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    print(f"Found {len(state['job_ids'])} matching jobs from DB.")
    placeholders = ",".join("?" * len(state["job_ids"]))
    query = f"SELECT link, role, company, skills, summary FROM jobs WHERE link IN ({placeholders})"
    cur.execute(query, state["job_ids"])
    rows = cur.fetchall()
    conn.close()
    return {"raw_jobs": [dict(row) for row in rows]}

def analyze_jobs_node(state: UnifiedState):
    """
    Iterates through jobs and uses LLM with structured output
    to calculate compatibility.
    """
    
    resume = state["resume_text"]
    jobs = state["raw_jobs"]
    results = []
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=open_ai_key, temperature=0.7)
    structured_llm = llm.with_structured_output(JobCompatibility)
    
    system_prompt = """You are an expert Career Coach. Compare the candidate's resume with the job description.
    1. Score the match from 0-100 based on skills and experience overlap.
    2. Identify critical missing skills.
    3. Write a concise, punchy message to the recruiter highlighting the candidate's strongest matching skill.
    """
    
    for job in jobs:
        user_prompt = f"RESUME: {resume}...\n\nJOB: {job['role']} at {job['company']}\nSummary: {job['summary']}\nSkills: {job['skills']}"
        try:
            msg = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            resp = structured_llm.invoke(msg)
            resp.apply_link = job['link']
            results.append(resp)
        except Exception:
            pass
            
    return {"analysis_results": results}

# --- 2. MASTER ROUTER & GRAPH ---

def master_router(state: UnifiedState):
    """Decides which workflow to trigger based on 'mode'."""
    print(f"--- Routing based on mode: {state['mode']} ---")
    if state["mode"] == "search":
        return "search"
    elif state["mode"] == "match":
        return "embed_resume"
    elif state["mode"] == "view":
        return "view_jobs"
    return END

workflow = StateGraph(UnifiedState)

# Nodes
workflow.add_node("search", search_node)
workflow.add_node("scrape", scrape_node)
workflow.add_node("extract", extraction_node)
workflow.add_node("save_to_db", save_new_jobs)
workflow.add_node("save_to_vector_db", embed_and_store_jobs)


workflow.add_node("embed_resume", embed_resume)

workflow.add_node("view_jobs", view_jobs_node)


workflow.add_node("retrieve_jobs", retrieve_similar_jobs)
workflow.add_node("load_structured", load_jobs_for_matching)
workflow.add_node("analyze_jobs", analyze_jobs_node)

# Entry Point (Router)
workflow.set_entry_point("router_node")

# Special node just for routing decision (identity node usually, but we can do this via conditional_entry too)
# Actually, let's use a dummy start node that leads to conditional edges
def start_node(state): return state
workflow.add_node("router_node", start_node)

workflow.add_conditional_edges(
    "router_node",
    master_router,
    {
        "search": "search",
        "embed_resume": "embed_resume",
        "view_jobs": "view_jobs"
    }
)

# Search Edges
workflow.add_edge("search", "scrape")
workflow.add_edge("scrape", "extract")
workflow.add_edge("extract", "save_to_db")
workflow.add_edge("save_to_db", "save_to_vector_db")
workflow.add_edge("save_to_vector_db", END)

# View Edges
workflow.add_edge("view_jobs", END)

# Match Edges
workflow.add_edge("embed_resume", "retrieve_jobs")
workflow.add_edge("retrieve_jobs", "load_structured")
workflow.add_edge("load_structured", "analyze_jobs")
workflow.add_edge("analyze_jobs", END)

app_graph = workflow.compile()

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI Job Agent", layout="wide")

st.title("🤖 AI Job Hunter & Matcher Agent")

# Sidebar for Navigation
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Select Mode", ["Search Jobs", "Analyze Resume", "View Database"])

# Initialize session state for displaying results
if "job_results" not in st.session_state: st.session_state.job_results = []
if "analysis_results" not in st.session_state: st.session_state.analysis_results = []

# --- MODE: SEARCH JOBS ---
if mode == "Search Jobs":
    st.header("🔍 Search for New Jobs")
    query = st.text_area("Enter Search Query (Boolean string works best)", 
                         value='"Data Scientist" AND "Python" AND "Remote"')
    
    if st.button("Start AI Search Agent"):
        with st.spinner("Agent is searching Google, scraping sites, and extracting data..."):
            inputs = {"query": query, "db_path": DB_PATH, "mode": "search"}
            result = app_graph.invoke(inputs)
            
            new_count = result.get("new_jobs_count", 0)
            st.success(f"Search Complete! Added {new_count} new unique jobs to the database.")
            
            # Show the newly found jobs
            if result.get("extracted_jobs"):
                st.subheader("Jobs Found in this Run:")
                for job in result["extracted_jobs"]:
                    with st.expander(f"{job.role} at {job.company}"):
                        st.write(f"**Experience:** {job.experience}")
                        st.write(f"**Skills:** {', '.join(job.skills)}")
                        st.write(job.summary)
                        st.markdown(f"[Apply Here]({job.link})")

# --- MODE: ANALYZE RESUME ---
elif mode == "Analyze Resume":
    st.header("🎯 Match Resume to Saved Jobs")
    
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Analyze Compatibility"):
            with st.spinner("Reading resume, querying vector DB, and analyzing matches..."):
                # Save uploaded file to temp path because Docling needs a path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                inputs = {
                    "resume_path": tmp_path, 
                    "db_path": DB_PATH, 
                    "mode": "match"
                }
                
                result = app_graph.invoke(inputs)
                st.session_state.analysis_results = result.get("analysis_results", [])
                
                # Cleanup temp file
                os.remove(tmp_path)

    # Display Results
    if st.session_state.analysis_results:
        st.subheader("Top Matches found in Database")
        for match in st.session_state.analysis_results:
            color = "green" if match.compatibility_score > 75 else "orange" if match.compatibility_score > 50 else "red"
            st.markdown(f"### {match.role} @ {match.company} - :{color}[{match.compatibility_score}% Match]")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Reasoning:** {match.reasoning}")
                st.info(f"**Draft Recruiter Msg:** {match.recruiter_message}")
            with col2:
                st.write("**Missing Skills:**")
                for skill in match.missing_skills:
                    st.error(skill)
                st.markdown(f"[👉 Apply Link]({match.apply_link})")
            st.divider()

# --- MODE: VIEW DATABASE ---
elif mode == "View Database":
    st.header("💾 Saved Jobs Database")
    if st.button("Refresh Data"):
        inputs = {"mode": "view", "db_path": DB_PATH}
        result = app_graph.invoke(inputs)
        st.session_state.job_results = result.get("db_jobs", [])

    if st.session_state.job_results:
        # Convert list of dicts to dataframe for nicer display
        import pandas as pd
        df = pd.DataFrame(st.session_state.job_results)
        if not df.empty:
            st.dataframe(df[['role', 'company', 'job_posted_date', 'skills', 'link', 'summary']], use_container_width=True)
        else:
            st.warning("Database is empty.")