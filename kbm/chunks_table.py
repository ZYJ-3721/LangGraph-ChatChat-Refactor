from kbm.base import Base, with_session
from kbm.files_table import FilesTable

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy import select, update
from sqlalchemy.orm import relationship
from sqlalchemy.event import listens_for


class ChunksTable(Base):
    """文本块信息映射表"""
    __tablename__ = "Chunks"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="文本块ID")
    kb_id = Column(Integer, ForeignKey("KBS.id"), nullable=False, comment="所属知识库ID") # 设置外键约束
    file_id = Column(Integer, ForeignKey("Files.id"), nullable=False, comment="所属文件ID") # 设置外键约束
    vector_id = Column(Integer, nullable=False, comment="文本块向量ID") # 用于查询向量数据库
    create_time = Column(DateTime, default=func.current_timestamp(), comment="创建时间")
    update_time = Column(DateTime, default=func.current_timestamp(), comment="更新时间", onupdate=func.current_timestamp())
    UniqueConstraint(vector_id, file_id, kb_id, name="unique_vector_id_file_id_kb_id") # 设置联合唯一性约束
    file = relationship("FilesTable", back_populates="chunks") # 设置与FilesTable表的一对多关系

    def __repr__(self):
        return f'ChunksTable(id={self.id}, kb_id={self.kb_id}, file_id={self.file_id}, vector_id={self.vector_id}, create_time={self.create_time}, update_time={self.update_time})'


@listens_for(ChunksTable, 'after_insert') # 监听插入事件
@listens_for(ChunksTable, 'after_delete') # 监听删除事件
def update_chunk_count(mapper, connection, target): # 更新文本块数量
    select_sql = select(func.count(ChunksTable.id)).where(ChunksTable.file_id==target.file_id)
    chunk_count = connection.execute(select_sql).scalar() # 统计事件发生后对应id的文本块数量
    update_sql = update(FilesTable).where(FilesTable.id==target.file_id).values(chunk_count=chunk_count)
    connection.execute(update_sql) # 执行sql语句：更新FilesTable表中对应id的chunk_count字段


@with_session
def add_chunks_to_db(session, kb_id: int, file_id: int, vector_ids: list[int]):
    """添加文本块信息到数据库表中"""
    new_chunk_list = []
    for vector_id in vector_ids:
        chunk = ChunksTable(kb_id=kb_id, file_id=file_id, vector_id=vector_id)
        new_chunk_list.append(chunk)
    session.add_all(new_chunk_list)

@with_session
def get_chunks_from_db(session, kb_id: int, file_id: int):
    """从数据库表中获取文本块信息"""
    chunks = session.query(ChunksTable).filter(
        ChunksTable.file_id.ilike(file_id),
        ChunksTable.kb_id.ilike(kb_id)).all()
    return [{
        "id": chunk.id,
        "kb_id": chunk.kb_id,
        "file_id": chunk.file_id,
        "vector_id": chunk.vector_id,
        "create_time": chunk.create_time,
        "update_time": chunk.update_time,
    } for chunk in chunks] # 返回字典列表

@with_session
def delete_chunks_from_db(session, kb_id: int, file_id: int, vector_ids: list[int]):
    """从数据库表中删除文本块信息"""
    existing_chunks = session.query(ChunksTable).filter(
        ChunksTable.vector_id.in_(vector_ids),
        ChunksTable.file_id.ilike(file_id),
        ChunksTable.kb_id.ilike(kb_id)).all()
    for chunk in existing_chunks:
        session.delete(chunk)

@with_session
def get_vector_ids(session, kb_id: int, file_id: int):
    """获取指定知识库和文件的所有文本块的向量ID"""
    chunks = session.query(ChunksTable).filter(
        ChunksTable.file_id.ilike(file_id),
        ChunksTable.kb_id.ilike(kb_id)).all()
    return [chunk.id for chunk in chunks]
