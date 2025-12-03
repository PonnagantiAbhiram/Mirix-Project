from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Text, DateTime, JSON, func
import uuid
from datetime import datetime, timedelta
from langchain.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType

Base = declarative_base()

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class EpisodicEvent(Base):
    __tablename__ = "episodic_memory"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)

    event_type = Column(String)
    actor = Column(String)
    summary = Column(Text)
    details = Column(Text)

    occurred_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    last_modified = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    summary_embedding = Column(JSON, nullable=True)
    details_embedding = Column(JSON, nullable=True)
    metadata_ = Column(JSON, nullable=True)


# ---------------------------------------------------------
# MEMORY MANAGER
# ---------------------------------------------------------
class EpisodicMemoryManager:

    def __init__(self, session_maker):
        self.session_maker = session_maker

    # ADD OR UPDATE MEMORY
    def add_or_update(
        self,
        user_id,
        event_type=None,
        actor=None,
        summary=None,
        details=None,
        event_id=None,
        occurred_at=None
    ):
        with self.session_maker() as session:

            # UPDATE MEMORY
            if event_id:
                event = session.query(EpisodicEvent).filter_by(id=event_id).first()
                if not event:
                    return f"No memory found with ID: {event_id}"

                if summary:
                    event.summary = summary
                if details:
                    event.details = (event.details or "") + "\n" + details
                if event_type:
                    event.event_type = event_type
                if actor:
                    event.actor = actor

                event.last_modified = datetime.utcnow()
                session.commit()
                session.refresh(event)
                return f"Memory {event_id} updated successfully."

            # ADD NEW MEMORY
            occurred_at = occurred_at or datetime.utcnow()

            event = EpisodicEvent(
                user_id=user_id,
                event_type=event_type,
                actor=actor,
                summary=summary,
                details=details,
                occurred_at=occurred_at
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event.id

    # LIST MEMORY
    def list_memories(self, user_id, limit=50):
        with self.session_maker() as session:
            return (
                session.query(EpisodicEvent)
                .filter(EpisodicEvent.user_id == user_id)
                .order_by(EpisodicEvent.occurred_at.desc())
                .limit(limit)
                .all()
            )

    # SEARCH MEMORY (retrive)
    def retrive(self, user_id, query):
        query = query.lower()
        with self.session_maker() as session:
            return (
                session.query(EpisodicEvent)
                .filter(
                    (EpisodicEvent.user_id == user_id)
                    & (
                        EpisodicEvent.summary.ilike(f"%{query}%")
                        | EpisodicEvent.details.ilike(f"%{query}%")
                        | EpisodicEvent.event_type.ilike(f"%{query}%")
                        | EpisodicEvent.actor.ilike(f"%{query}%")
                    )
                )
                .all()
            )

    # GET EVENTS IN RANGE
    def get_events_in_range(self, user_id, days=1):
        now = datetime.utcnow()
        start = now - timedelta(days=days)

        with self.session_maker() as session:
            return (
                session.query(EpisodicEvent)
                .filter(EpisodicEvent.user_id == user_id)
                .filter(EpisodicEvent.occurred_at >= start)
                .order_by(EpisodicEvent.occurred_at.desc())
                .all()
            )


# ---------------------------------------------------------
# FORMAT MEMORY OUTPUT
# ---------------------------------------------------------
def format_event(event):

    def nice_time(dt):
        if not dt:
            return "Unknown time"
        return dt.strftime("%b %d, %Y — %I:%M %p")

    lines = []
    lines.append(f"🕒 **{nice_time(event.occurred_at)}**")
    if event.event_type:
        lines.append(f"📌 **Event Type:** {event.event_type}")
    if event.actor:
        lines.append(f"🧑 **Actor:** {event.actor}")
    lines.append(f"📝 **Summary:** {event.summary}")
    if event.details:
        lines.append(f"📄 **Details:** {event.details}")
    return "\n".join(lines)


# ---------------------------------------------------------
# DATABASE INIT
# ---------------------------------------------------------
engine = create_engine("sqlite:///episodic_memory.db")
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)
memory_manager = EpisodicMemoryManager(SessionLocal)


# ---------------------------------------------------------
# TOOLS (FINAL — THIS WILL WORK)
# ---------------------------------------------------------

@tool("add_episodic_memory", return_direct=True)
def add_episodic_memory(
    user_id: str,
    event_type: str = None,
    actor: str = None,
    summary: str = None,
    details: str = None
):
    """Add a new episodic memory."""
    memory_id = memory_manager.add_or_update(
        user_id=user_id,
        event_type=event_type,
        actor=actor,
        summary=summary,
        details=details
    )
    return f"Memory stored with ID: {memory_id}"


@tool("search_memory", return_direct=True)
def search_memory(user_id: str, query: str):
    """Search episodic memories."""
    results = memory_manager.retrive(user_id, query)
    return [format_event(e) for e in results]


@tool("list_recent_events", return_direct=True)
def list_recent_events(user_id: str, days: int = 7):
    """List recent events."""
    events = memory_manager.get_events_in_range(user_id, days)
    return [format_event(e) for e in events]


@tool("list_memory", return_direct=True)
def list_memory(user_id: str):
    """List all stored memories."""
    events = memory_manager.list_memories(user_id)
    return [format_event(e) for e in events]


@tool("update_episodic_memory", return_direct=True)
def update_episodic_memory(event_id: str, new_summary: str = None, new_details: str = None):
    """Update a stored memory."""
    result = memory_manager.add_or_update(
        user_id="user_123",
        event_id=event_id,
        summary=new_summary,
        details=new_details
    )
    return result


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
system_prompt = """
You are an Episodic Memory Agent.
Use user_id "user_123" for all operations.

Rules:
- New event → add_episodic_memory
- Keyword search → search_memory
- Recent events → list_recent_events
- All events → list_memory
- Update → update_episodic_memory

When showing memory results:
- Use format_event EXACTLY
- Separate events using ---
"""


# ---------------------------------------------------------
# AGENT INIT
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyBbjqlSWwg67qoJiukd8eYt9qG7KGdEHEU"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

tools = [
    add_episodic_memory,
    search_memory,
    list_memory,
    list_recent_events,
    update_episodic_memory
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs={"system_message": system_prompt},
    verbose=True
)


# ---------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------
print(agent.invoke({"input": "I met my friend Arjun today and we ate biryani."}))
print(agent.invoke({"input": "I had a meeting with the product team yesterday about the new app launch."}))
print(agent.invoke({"input": "What happened today?"}))
print(agent.invoke({"input": "Search my memories about Arjun."}))
print(agent.invoke({"input": "List all my recent events."}))

