from crewai import Agent, Task, Crew, Process, LLM
from src.config import OLLAMA_HOST, SUMMARIZER_MODEL, QA_MODEL, MCQ_MODEL
from src.vectorstore import get_store, retriever
from src.config import PROMPTS_DIR, QA_K


# Initialize LLMs for CrewAI agents (using CrewAI native LLM wrapper)
summarizer_llm = LLM(model=f"ollama/{SUMMARIZER_MODEL}", base_url=OLLAMA_HOST, temperature=0.2)
qa_llm = LLM(model=f"ollama/{QA_MODEL}", base_url=OLLAMA_HOST, temperature=0.1)
mcq_llm = LLM(model=f"ollama/{MCQ_MODEL}", base_url=OLLAMA_HOST, temperature=0.4)


# Define CrewAI Agents
researcher_agent = Agent(
    role="Financial Document Researcher",
    goal="Extract and analyze relevant information from financial documents and APIs",
    backstory=(
        "You are an expert financial analyst with deep knowledge of financial statements, "
        "market data, and corporate filings. You excel at finding specific information "
        "from large document collections and presenting it clearly."
    ),
    llm=qa_llm,
    verbose=False,
    allow_delegation=False
)

summarizer_agent = Agent(
    role="Financial Document Summarizer",
    goal="Create comprehensive summaries of financial documents that preserve key metrics and insights",
    backstory=(
        "You are a financial summarization specialist who can distill complex financial "
        "documents into clear, actionable summaries. You preserve critical numbers, dates, "
        "and forward-looking statements while removing redundancy."
    ),
    llm=summarizer_llm,
    verbose=False,
    allow_delegation=False
)

insight_generator_agent = Agent(
    role="Financial Insight Generator",
    goal="Generate actionable insights, trends, and educational content from financial data",
    backstory=(
        "You are a financial educator and analyst who creates valuable insights from data. "
        "You can identify trends, generate quiz questions for learning, and highlight "
        "important patterns in financial information."
    ),
    llm=mcq_llm,
    verbose=False,
    allow_delegation=False
)


def create_research_task(question: str, doc_id: str | None = None) -> Task:
    """Create a research task for the researcher agent."""
    # Retrieve relevant context
    ret = retriever(k=QA_K, doc_id=doc_id)
    docs = ret.invoke(question)

    context_parts = []
    for doc in docs:
        source = doc.metadata["source"]
        if "page" in doc.metadata:
            page = doc.metadata["page"]
            context_parts.append(f"[source: {source} p.{page}]\n{doc.page_content}")
        else:
            row = doc.metadata.get("row", "N/A")
            context_parts.append(f"[source: {source} row {row}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt_template = (PROMPTS_DIR / "qa.txt").read_text()
    description = prompt_template.format(context=context, question=question)

    return Task(
        description=description,
        expected_output="A clear, factual answer to the question with source citations",
        agent=researcher_agent
    )


def create_summarize_task(doc_id: str | None = None) -> Task:
    """Create a summarization task for the summarizer agent."""
    store = get_store()

    if doc_id:
        result = store.get(where={"doc_id": doc_id})
        if not result["ids"]:
            raise ValueError(f"No chunks found for doc_id: {doc_id}")
    else:
        result = store.get()
        if not result["ids"]:
            raise ValueError("No documents found in the database")

    chunks = result["documents"]
    content = "\n\n".join(chunks[:10])

    description = (
        "Produce a faithful, dense summary of the financial content below. Preserve:\n"
        "- Key figures (revenue, profit, growth rates) with their units and periods\n"
        "- Named entities (companies, segments, executives)\n"
        "- Material risks, commitments, and forward-looking statements\n\n"
        "Synthesize into 7-12 bullet points grouped by theme.\n\n"
        f"Content:\n{content}"
    )

    return Task(
        description=description,
        expected_output="A comprehensive summary with 7-12 bullet points covering key financial metrics, entities, and insights",
        agent=summarizer_agent
    )


def create_insight_task(topic: str, doc_id: str | None = None, n: int = 5) -> Task:
    """Create an insight generation task (MCQ generation)."""
    from src.config import MCQ_K

    ret = retriever(k=MCQ_K, doc_id=doc_id)
    docs = ret.invoke(topic)

    context_parts = [doc.page_content for doc in docs]
    context = "\n\n".join(context_parts)

    prompt_template = (PROMPTS_DIR / "mcq.txt").read_text()
    description = prompt_template.format(n=n, topic=topic, context=context)

    return Task(
        description=description,
        expected_output=f"Exactly {n} multiple-choice questions in JSON format with 4 options each, correct answer index, and explanations",
        agent=insight_generator_agent
    )


def run_crew_workflow(workflow_type: str, **kwargs) -> dict:
    """
    Run a CrewAI workflow for document analysis.

    Args:
        workflow_type: Type of workflow ('research', 'summarize', 'insights')
        **kwargs: Additional parameters based on workflow type

    Returns:
        Dictionary with workflow results
    """
    tasks = []

    if workflow_type == "research":
        question = kwargs.get("question")
        doc_id = kwargs.get("doc_id")
        if not question:
            raise ValueError("Question required for research workflow")
        tasks.append(create_research_task(question, doc_id))

    elif workflow_type == "summarize":
        doc_id = kwargs.get("doc_id")
        tasks.append(create_summarize_task(doc_id))

    elif workflow_type == "insights":
        topic = kwargs.get("topic")
        doc_id = kwargs.get("doc_id")
        n = kwargs.get("n", 5)
        if not topic:
            raise ValueError("Topic required for insights workflow")
        tasks.append(create_insight_task(topic, doc_id, n))

    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    # Create and run the crew
    crew = Crew(
        agents=[researcher_agent, summarizer_agent, insight_generator_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()

    return {
        "workflow_type": workflow_type,
        "result": result,
        "tasks_completed": len(tasks)
    }
