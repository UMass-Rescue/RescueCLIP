import sqlalchemy
from sqlalchemy import BigInteger, Column, Float, Integer, String, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()

BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), "sqlite")


class ThresholdTest(Base):
    __tablename__ = "threshold_test"

    id = Column(BigIntegerType, primary_key=True)
    model_name = Column(String(200), unique=False, nullable=False)
    threshold = Column(Float, unique=False, nullable=False)
    tp = Column(Integer, unique=False, nullable=False)
    fp = Column(Integer, unique=False, nullable=False)
    tn = Column(Integer, unique=False, nullable=False)
    fn = Column(Integer, unique=False, nullable=False)


def try_fetch_by_field(cls, field, target, session):
    try:
        return session.execute(select(cls).where(getattr(cls, field) == target))
    except TypeError:
        return None


def try_insert(entry: object, session: Session):
    """performs an insert into one of the db tables

    Args:
        entry (obj): object of one of the above 5 classes
        session (obj): sqlite Session
    Return:
        return the entry that now includes the id
    """
    try:
        session.add(entry)
        session.commit()
        return entry
    except IntegrityError:
        session.rollback()
        return None


def initialize(database_url) -> Session:
    """Create tables if needed.

    Args:
        database_url (string): URL for database
    Return:
        session (Session): access to sql databse
    """
    # Establish a database connection
    # Create an engine to connect to a SQLite database
    engine = sqlalchemy.create_engine(database_url)

    # check if db tables exist:
    insp = sqlalchemy.inspect(engine)
    if not insp.has_table("Image"):
        # initialize db with empty tables
        Base.metadata.create_all(engine)

    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    return Session()
