import os
import json
from functools import wraps
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base

KBS_ROOT = os.path.join(os.getcwd(), "data", "kbs")

os.makedirs(KBS_ROOT, exist_ok=True)


# 声明一个映射表基类
Base: DeclarativeMeta = declarative_base()
# 创建一个数据库会话
engine = create_engine(
    url=f"sqlite:///{KBS_ROOT}/info.db", # 默认使用sqlite数据库
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False))
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def session_scope():
    """上下文管理器用于自动管理session"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def with_session(func):
    """装饰器用于额外添加session"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            return func(session, *args, **kwargs)
    return wrapper

def create_tables():
    """创建数据库表"""
    Base.metadata.create_all(bind=engine)
