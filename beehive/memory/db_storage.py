import contextlib
import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# SQLAlchemy / SQLModel imports
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.sql.base import Executable
from sqlmodel import JSON, Column, Field, SQLModel

from beehive.constants import INTERNAL_FOLDER_PATH
from beehive.invokable.types import AnyMessage
from beehive.message import BHMessage, BHToolMessage, MessageRole

########## Constants ##########

SQLLITE_DB_URI = f"sqlite:///{Path(INTERNAL_FOLDER_PATH).resolve()}/beehive.db"


########## Models ##########


def create_unique_id() -> str:
    return str(uuid4())


class BeehiveModel(SQLModel, table=True):
    __tablename__ = "beehives"

    id: str = Field(default_factory=create_unique_id, primary_key=True)
    name: str = Field(primary_key=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    task: str = Field()


class TaskModel(SQLModel, table=True):
    __tablename__ = "tasks"

    id: str = Field(default_factory=create_unique_id, primary_key=True)
    content: str = Field()
    invokable: str = Field(foreign_key="invokables.name")
    beehive: str | None = Field(foreign_key="beehives.id", default=None)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class InvokableModel(SQLModel, table=True):
    __tablename__ = "invokables"

    name: str = Field(primary_key=True)
    type: str = Field()
    tools: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    backstory: str = Field(default="You are a helpful AI assistant.")


class MessageModel(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(default_factory=create_unique_id, primary_key=True)
    task: str = Field(foreign_key="tasks.id")
    content: str = Field()
    role: str = Field()
    message_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class ToolCallModel(SQLModel, table=True):
    __tablename__ = "tool_calls"

    tool_call_id: str = Field(primary_key=True)
    message: str = Field(foreign_key="messages.id")
    name: str = Field()
    args: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ToolMessageModel(SQLModel, table=True):
    __tablename__ = "tool_messages"
    id: str = Field(default_factory=create_unique_id, primary_key=True)
    tool_call_id: str = Field(foreign_key="tool_calls.tool_call_id")
    content: str


########## SessionFactory class ##########


class SessionFactory:
    db_uri: str
    engine: Engine

    def __init__(self, db_uri: str, engine: Engine):
        self.db_uri = db_uri
        self.engine = engine

    @contextlib.contextmanager
    def create_session(self):
        session_factory = sessionmaker()
        Session = scoped_session(session_factory)
        Session.configure(bind=self.engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def execute_stmt(
        self,
        stmt: Executable,
        session: Session,
        select_statement: bool = True,
        model_objects: bool = True,
    ) -> Sequence[Any]:
        if select_statement:
            if model_objects:
                result = session.scalars(stmt).all()
            else:
                result = session.execute(stmt).all()
            return result
        else:
            session.execute(stmt)
            return []


def setup(db_uri: str, engine: Engine) -> None:
    db_factory = SessionFactory(
        db_uri=db_uri,
        engine=engine,
    )
    SQLModel.metadata.create_all(bind=db_factory.engine)
    return None


########## Storage class ##########


class DbStorage:
    db_uri: str
    engine: Engine

    def __init__(self, db_uri: str | None = None):
        self.db_uri = SQLLITE_DB_URI if not db_uri else db_uri
        self.engine = create_engine(self.db_uri)
        setup(self.db_uri, self.engine)

    def get_model_objects(
        self, stmt: Executable, model_objects: bool = True
    ) -> Sequence[Any]:
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            res = factory.execute_stmt(stmt, session, True, model_objects)
        return res

    def add_task(
        self,
        task: str,
        invokable: "Invokable",  # type: ignore # noqa: F821
    ) -> str:
        task_id = str(uuid4())
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            task_obj = TaskModel(id=task_id, content=task, invokable=invokable.name)
            session.add(task_obj)
            session.commit()
        return task_id

    def add_task_to_beehive(self, task_id: str, beehive_id: str) -> str:
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            res = factory.execute_stmt(stmt, session)
            if not res:
                raise ValueError(f"Could not find task with ID {task_id}!")
            task_obj = res[0]
            if not isinstance(task_obj, TaskModel):
                raise ValueError(
                    f"Incorrect return type. Expecting `TaskModel`, found `{task_obj.__class__.__name__}`."
                )
            task_obj.beehive = beehive_id
            session.commit()
        return task_id

    def add_beehive(
        self,
        invokable: "Invokable",  # type: ignore # noqa: F821
        task: str,
    ) -> str:
        bh_id = str(uuid4())
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            bh_obj = BeehiveModel(id=bh_id, name=invokable.name, task=task)
            session.add(bh_obj)
            session.commit()
        return bh_id

    def add_invokable(self, invokable: "Invokable") -> str:  # type: ignore # noqa: F821
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            stmt = select(InvokableModel).where(InvokableModel.name == invokable.name)
            res = factory.execute_stmt(stmt, session)
            if len(res) == 0:
                tools_list: list[dict[str, Any]] = []
                if hasattr(invokable, "_tools_serialized"):
                    if invokable._tools_serialized:
                        for _, serialized in invokable._tools_serialized.items():
                            tools_list.append(serialized)

                invokable_obj = InvokableModel(
                    name=invokable.name,
                    backstory=invokable.backstory,
                    type=invokable.__class__.__name__,
                    tools=tools_list,
                )
                session.add(invokable_obj)
                session.commit()

        # For mypy
        if not isinstance(invokable.name, str):
            raise ValueError("Invokable's name is not a string!")
        return invokable.name

    def add_message(
        self,
        task_id: str,
        message: AnyMessage,
    ):
        # DB ID
        message_id = str(uuid4())

        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            res = factory.execute_stmt(stmt, session, True, True)
            assert res, print("Could not find task!")

            if isinstance(message, BHMessage):
                message_obj = MessageModel(
                    id=message_id,
                    task=task_id,
                    role=message.role,
                    content=message.content,
                )
                session.add(message_obj)

            # Since messages are added sequentially, ToolMessages should get added after
            # the associated ToolCall object.
            elif isinstance(message, BHToolMessage):
                tool_message_obj = ToolMessageModel(
                    tool_call_id=message.tool_call_id, content=message.content
                )
                session.add(tool_message_obj)

            elif isinstance(message, BaseMessage):
                # As above, messages are added sequentially. Therefore, ToolMessages
                # should get added after the associated ToolCall object.
                if isinstance(message, ToolMessage):
                    tool_message_obj = ToolMessageModel(
                        tool_call_id=message.tool_call_id, content=message.content
                    )
                    session.add(tool_message_obj)
                else:
                    if isinstance(message, AIMessage):
                        role = MessageRole.ASSISTANT
                    elif isinstance(message, HumanMessage):
                        role = MessageRole.USER
                    elif isinstance(message, SystemMessage):
                        role = MessageRole.SYSTEM
                    else:
                        raise ValueError(
                            f"Unrecognized message class `{message.__class__.__name__}`"
                        )

                    message_obj = MessageModel(
                        id=message_id,
                        task=task_id,
                        role=role,
                        content=message.content,
                        message_metadata=message.response_metadata,
                    )
                    session.add(message_obj)
            session.commit()

        return message_id

    def add_tool_calls(
        self,
        message: BHMessage | BaseMessage,
        db_message_id: str,
    ) -> None:
        factory = SessionFactory(self.db_uri, self.engine)
        with factory.create_session() as session:
            # Make sure message exists in database
            stmt = select(MessageModel).where(MessageModel.id == db_message_id)
            res = factory.execute_stmt(stmt, session)
            if not res:
                raise ValueError(
                    f"Could not find message object with id `{db_message_id}`!"
                )

            if isinstance(message, BHMessage):
                for bh_tc in message.tool_calls:
                    tc_obj = ToolCallModel(
                        tool_call_id=bh_tc.tool_call_id,
                        message=db_message_id,
                        name=bh_tc.tool_name,
                        args=bh_tc.tool_arguments,
                    )
                    session.add(tc_obj)

            # Add tool calls as well. Only Langchain AIMessage's have tool
            # calls.
            elif isinstance(message, AIMessage):
                for lc_tc in message.tool_calls:
                    tc_obj = ToolCallModel(
                        tool_call_id=lc_tc["id"],
                        message=db_message_id,
                        name=lc_tc["name"],
                        args=lc_tc["args"],
                    )
                    session.add(tc_obj)

            session.commit()
        return None
