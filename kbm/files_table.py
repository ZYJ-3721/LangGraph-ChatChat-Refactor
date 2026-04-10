from kbm.base import Base, with_session
from kbm.kbs_table import KBSTable

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy import select, update
from sqlalchemy.orm import relationship
from sqlalchemy.event import listens_for


class FilesTable(Base):
    """文件信息映射表"""
    __tablename__ = "Files"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="文件ID")
    file_name = Column(String(200), nullable=False, comment="文件名称")
    file_type = Column(String(10), nullable=False, comment="文件类型")
    file_size = Column(Integer, nullable=False, comment="文件大小")
    file_dir = Column(String(200), nullable=False, comment="文件目录")
    kb_id = Column(Integer, ForeignKey("KBS.id"), nullable=False, comment="所属知识库ID") # 设置外键约束
    loader_name = Column(String(100), nullable=False, comment="文件加载器名称")
    splitter_name = Column(String(100), nullable=False, comment="文本分割器名称")
    chunk_count = Column(Integer, default=0, comment="文本块数量")
    create_time = Column(DateTime, default=func.current_timestamp(), comment="创建时间")
    update_time = Column(DateTime, default=func.current_timestamp(), comment="更新时间", onupdate=func.current_timestamp())
    UniqueConstraint(file_name, kb_id, name="unique_file_name_kb_id") # 设置联合唯一性约束
    kb = relationship("KBSTable", back_populates="files") # 设置与KBSTable表的一对多关系
    chunks = relationship("ChunksTable", back_populates="file", cascade="all, delete-orphan") # 设置与ChunksTable表的一对多关系

    def __repr__(self):
        return f'FilesTable(id={self.id}, file_name="{self.file_name}", file_type="{self.file_type}", file_size={self.file_size}, file_dir="{self.file_dir}", kb_id={self.kb_id}, loader_name="{self.loader_name}", splitter_name="{self.splitter_name}", chunk_count={self.chunk_count}, create_time={self.create_time}, update_time={self.update_time})'


@listens_for(FilesTable, 'after_insert') # 监听插入事件
@listens_for(FilesTable, 'after_delete') # 监听删除事件
def update_file_count(mapper, connection, target): # 更新文件数量
    select_sql = select(func.count(FilesTable.id)).where(FilesTable.kb_id==target.kb_id)
    file_count = connection.execute(select_sql).scalar() # 统计监听事件发生后对应id的文件数量
    update_sql = update(KBSTable).where(KBSTable.id==target.kb_id).values(file_count=file_count)
    connection.execute(update_sql) # 执行sql语句：更新KBSTable表中对应id的file_count字段


@with_session
def add_files_to_db(session, kb_id: int, file_infos: list[dict]):
    """添加文件信息到数据库表中"""
    file_names = [f["file_name"] for f in file_infos]
    existing_files = session.query(FilesTable).filter(
        FilesTable.file_name.in_(file_names),
        FilesTable.kb_id.ilike(kb_id)).all()
    existing_maps = {F.file_name: F for F in existing_files}
    new_file_list = []
    for file_info in file_infos:
        if file_info["file_name"] not in existing_maps: # 添加新实例
            file = FilesTable(kb_id=kb_id, **file_info)
            new_file_list.append(file)
        else: # 更新旧实例
            file = existing_maps[file_info["file_name"]]
            file.file_size = file_info["file_size"]
            file.loader_name = file_info["loader_name"]
            file.splitter_name = file_info["splitter_name"]
    session.add_all(new_file_list)

@with_session
def get_files_from_db(session, kb_id: int):
    """从数据库表中获取文件信息"""
    files = session.query(FilesTable).filter(FilesTable.kb_id.ilike(kb_id)).all()
    return [{
        "id": file.id,
        "file_name": file.file_name,
        "file_type": file.file_type,
        "file_size": file.file_size,
        "file_dir": file.file_dir,
        "kb_id": file.kb_id,
        "loader_name": file.loader_name,
        "splitter_name": file.splitter_name,
        "chunk_count": file.chunk_count,
        "create_time": file.create_time,
        "update_time": file.update_time,
    } for file in files] # 返回字典列表

@with_session
def delete_files_from_db(session, kb_id: int, file_names: list[str]):
    """从数据库表中删除文件信息"""
    existing_files = session.query(FilesTable).filter(
        FilesTable.file_name.in_(file_names),
        FilesTable.kb_id.ilike(kb_id)).all()
    for file in existing_files:
        session.delete(file)
