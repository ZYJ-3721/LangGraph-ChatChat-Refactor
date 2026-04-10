from kbm.base import Base, with_session

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship


class KBSTable(Base):
    """知识库信息映射表"""
    __tablename__ = "KBS"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="知识库ID")
    kb_name = Column(String(50), nullable=False, unique=True, comment="知识库名称")
    kb_type = Column(String(50), nullable=False, comment="向量库类型")
    kb_desc = Column(String(200), nullable=False, comment="知识库简介")
    platform = Column(String(50), nullable=False, comment="模型平台")
    api_url = Column(String(100), nullable=False, comment="API地址")
    api_key = Column(String(100), nullable=False, comment="API密钥")
    embedding_model = Column(String(50), nullable=False, comment="Embedding模型")
    file_count = Column(Integer, default=0, comment="文件数量")
    create_time = Column(DateTime, default=func.current_timestamp(), comment="创建时间")
    update_time = Column(DateTime, default=func.current_timestamp(), comment="更新时间", onupdate=func.current_timestamp())
    files = relationship("FilesTable", back_populates="kb", cascade="all, delete-orphan") # 设置与FilesTable表的一对多关系

    def __repr__(self):
        return f'KBSTable(id={self.id}, kb_name="{self.kb_name}", kb_type="{self.kb_type}", kb_desc="{self.kb_desc}", platform="{self.platform}", api_url="{self.api_url}", api_key="{self.api_key}", embedding_model="{self.embedding_model}", file_count={self.file_count}, create_time={self.create_time}, update_time={self.update_time})'


@with_session
def add_kb_to_db(session, kb_name, kb_type, kb_desc, platform, api_url, api_key, embedding_model):
    """添加知识库信息到数据库表中"""
    kb = session.query(KBSTable).filter(KBSTable.kb_name.ilike(kb_name)).first()
    if not kb: # 添加新实例
        kb = KBSTable(
            kb_name=kb_name, kb_type=kb_type, kb_desc=kb_desc,
            platform=platform, api_url=api_url, api_key=api_key,
            embedding_model=embedding_model)
        session.add(kb)
    else: # 更新旧实例
        kb.kb_desc = kb_desc
        kb.platform = platform
        kb.api_url = api_url
        kb.api_key = api_key

@with_session
def get_kb_from_db(session, kb_name):
    """从数据库表中获取知识库信息"""
    kb = session.query(KBSTable).filter(KBSTable.kb_name.ilike(kb_name)).first()
    return {
        "id": kb.id,
        "kb_name": kb.kb_name,
        "kb_type": kb.kb_type,
        "kb_desc": kb.kb_desc,
        "platform": kb.platform,
        "api_url": kb.api_url,
        "api_key": kb.api_key,
        "embedding_model": kb.embedding_model,
    }

@with_session
def delete_kb_from_db(session, kb_name):
    """从数据库表中删除知识库信息"""
    kb = session.query(KBSTable).filter(KBSTable.kb_name.ilike(kb_name)).first()
    session.delete(kb)

@with_session
def get_kb_names(session):
    """获取所有知识库名称"""
    return [kb_name[0] for kb_name in session.query(KBSTable.kb_name).all()]
