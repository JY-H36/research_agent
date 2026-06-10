"""
知识图谱 ORM 模型
8 张实体表 + 9 张关系表，继承 database.connection.Base
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum,
    Boolean, JSON, Float, PrimaryKeyConstraint, Index,
)
from sqlalchemy.orm import relationship
import enum

from database.connection import Base


# ============================================================
# 枚举类型
# ============================================================
class PaperSource(str, enum.Enum):
    UPLOAD = "upload"
    SEARCH = "search"


class MethodCategory(str, enum.Enum):
    FEATURE_EXTRACTOR = "feature_extractor"
    NETWORK_ARCHITECTURE = "network_architecture"
    LOSS_FUNCTION = "loss_function"


class MetricDirection(str, enum.Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"


class VenueType(str, enum.Enum):
    CONFERENCE = "conference"
    JOURNAL = "journal"
    WORKSHOP = "workshop"
    PREPRINT = "preprint"


class CitationType(str, enum.Enum):
    BACKGROUND = "background"
    METHOD_COMPARISON = "method_comparison"
    BASELINE = "baseline"
    USES_DATASET = "uses_dataset"


class PerformanceContribution(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ABLATION_ONLY = "ablation_only"


# ============================================================
# 辅助函数
# ============================================================
def _new_uuid() -> str:
    return uuid.uuid4().hex


# ============================================================
# 实体表
# ============================================================

class KgPaper(Base):
    """论文实体"""
    __tablename__ = "kg_papers"

    paper_id = Column(String(32), primary_key=True, default=_new_uuid)
    title = Column(Text, nullable=False)
    abstract = Column(Text, default="")
    abstract_cn = Column(Text, default="")
    authors = Column(JSON, default=list)
    keywords = Column(JSON, default=list)
    method_name = Column(String(500), default="")
    method_summary = Column(Text, default="")
    year = Column(Integer, nullable=True)
    doi = Column(String(255), default="")
    arxiv_id = Column(String(100), default="")
    url = Column(Text, default="")
    pdf_url = Column(Text, default="")
    venue_name = Column(String(500), default="")
    citation_count = Column(Integer, default=0)
    language = Column(String(10), default="en")
    source = Column(SQLEnum(PaperSource), default=PaperSource.UPLOAD)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    docling_md = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    paper_authors = relationship("KgPaperAuthor", back_populates="paper", cascade="all, delete-orphan")
    methods = relationship("KgPaperUsesMethod", back_populates="paper", cascade="all, delete-orphan")
    evaluations = relationship("KgPaperEvaluatesOn", back_populates="paper", cascade="all, delete-orphan")
    trainings = relationship("KgPaperTrainsOn", back_populates="paper", cascade="all, delete-orphan")
    tasks = relationship("KgPaperBelongsToTask", back_populates="paper", cascade="all, delete-orphan")
    metrics = relationship("KgPaperReportsMetric", back_populates="paper", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgPaper(id={self.paper_id}, title='{self.title[:60]}')>"


class KgAuthor(Base):
    """作者实体"""
    __tablename__ = "kg_authors"

    author_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(255), nullable=False, index=True)
    affiliation = Column(Text, default="")
    orcid = Column(String(100), default="")
    paper_count = Column(Integer, default=0)
    h_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    papers = relationship("KgPaperAuthor", back_populates="author", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgAuthor(id={self.author_id}, name='{self.name}')>"


class KgDataset(Base):
    """数据集实体"""
    __tablename__ = "kg_datasets"

    dataset_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(500), nullable=False, index=True)
    description = Column(Text, default="")
    domain = Column(String(100), default="")
    size = Column(String(100), default="")
    task = Column(String(255), default="")
    url = Column(Text, default="")
    year = Column(Integer, nullable=True)
    license = Column(String(255), default="")
    languages = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluations = relationship("KgPaperEvaluatesOn", back_populates="dataset", cascade="all, delete-orphan")
    trainings = relationship("KgPaperTrainsOn", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgDataset(id={self.dataset_id}, name='{self.name}')>"


class KgMethod(Base):
    """方法/模型实体"""
    __tablename__ = "kg_methods"

    method_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(500), nullable=False, index=True)
    aliases = Column(JSON, default=list)
    category = Column(String(50), nullable=True)
    description = Column(Text, default="")
    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="SET NULL"), nullable=True)
    year = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    used_by = relationship("KgPaperUsesMethod", back_populates="method", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgMethod(id={self.method_id}, name='{self.name}')>"


class KgMetric(Base):
    """评价指标实体"""
    __tablename__ = "kg_metrics"

    metric_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(500), default="")
    description = Column(Text, default="")
    direction = Column(SQLEnum(MetricDirection), nullable=True)
    unit = Column(String(50), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    reports = relationship("KgPaperReportsMetric", back_populates="metric", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgMetric(id={self.metric_id}, name='{self.name}')>"


class KgTask(Base):
    """研究任务实体（树形结构）"""
    __tablename__ = "kg_tasks"

    task_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(500), nullable=False, unique=True, index=True)
    description = Column(Text, default="")
    parent_task_id = Column(String(32), ForeignKey("kg_tasks.task_id", ondelete="SET NULL"), nullable=True)
    level = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    papers = relationship("KgPaperBelongsToTask", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KgTask(id={self.task_id}, name='{self.name}')>"


class KgVenue(Base):
    """发表 venue"""
    __tablename__ = "kg_venues"

    venue_id = Column(String(32), primary_key=True, default=_new_uuid)
    name = Column(String(500), nullable=False)
    abbreviation = Column(String(100), default="")
    type = Column(SQLEnum(VenueType), nullable=True)
    rank = Column(String(50), default="")
    publisher = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_venue_name", "name"),
    )

    def __repr__(self):
        return f"<KgVenue(id={self.venue_id}, name='{self.name}')>"


# ============================================================
# 关系表
# ============================================================

class KgPaperAuthor(Base):
    """论文-作者关系 (WRITTEN_BY)"""
    __tablename__ = "kg_paper_author"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    author_id = Column(String(32), ForeignKey("kg_authors.author_id", ondelete="CASCADE"), nullable=False)
    author_order = Column(Integer, default=1)
    is_corresponding = Column(Boolean, default=False)

    paper = relationship("KgPaper", back_populates="paper_authors")
    author = relationship("KgAuthor", back_populates="papers")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "author_id"),
    )

    def __repr__(self):
        return f"<KgPaperAuthor(paper={self.paper_id[:8]}, author={self.author_id[:8]})>"


class KgPaperCites(Base):
    """论文引用关系 (CITES)"""
    __tablename__ = "kg_paper_cites"

    citing_paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    cited_paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    context = Column(Text, default="")
    section = Column(String(255), default="")
    citation_type = Column(SQLEnum(CitationType), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("citing_paper_id", "cited_paper_id"),
    )

    def __repr__(self):
        return f"<KgPaperCites({self.citing_paper_id[:8]}→{self.cited_paper_id[:8]})>"


class KgPaperUsesMethod(Base):
    """论文使用方法关系 (USES_METHOD)"""
    __tablename__ = "kg_paper_uses_method"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    method_id = Column(String(32), ForeignKey("kg_methods.method_id", ondelete="CASCADE"), nullable=False)
    role = Column(Text, default="")
    context = Column(Text, default="")
    variant = Column(String(255), default="")
    performance_contribution = Column(SQLEnum(PerformanceContribution), nullable=True)

    paper = relationship("KgPaper", back_populates="methods")
    method = relationship("KgMethod", back_populates="used_by")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "method_id"),
    )

    def __repr__(self):
        return f"<KgPaperUsesMethod(paper={self.paper_id[:8]}, method={self.method_id[:8]})>"


class KgPaperEvaluatesOn(Base):
    """论文在数据集上评测关系 (EVALUATES_ON)"""
    __tablename__ = "kg_paper_evaluates_on"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(String(32), ForeignKey("kg_datasets.dataset_id", ondelete="CASCADE"), nullable=False)
    task = Column(String(255), default="")
    split = Column(String(50), default="eval")
    metrics = Column(JSON, default=dict)
    protocol = Column(String(255), default="")

    paper = relationship("KgPaper", back_populates="evaluations")
    dataset = relationship("KgDataset", back_populates="evaluations")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "dataset_id"),
    )

    def __repr__(self):
        return f"<KgPaperEvaluatesOn(paper={self.paper_id[:8]}, dataset={self.dataset_id[:8]})>"


class KgPaperTrainsOn(Base):
    """论文在数据集上训练关系 (TRAINS_ON)"""
    __tablename__ = "kg_paper_trains_on"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(String(32), ForeignKey("kg_datasets.dataset_id", ondelete="CASCADE"), nullable=False)
    split = Column(String(50), default="train+dev")

    paper = relationship("KgPaper", back_populates="trainings")
    dataset = relationship("KgDataset", back_populates="trainings")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "dataset_id"),
    )

    def __repr__(self):
        return f"<KgPaperTrainsOn(paper={self.paper_id[:8]}, dataset={self.dataset_id[:8]})>"


class KgPaperBelongsToTask(Base):
    """论文属于研究任务关系 (BELONGS_TO)"""
    __tablename__ = "kg_paper_belongs_to_task"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    task_id = Column(String(32), ForeignKey("kg_tasks.task_id", ondelete="CASCADE"), nullable=False)
    is_primary = Column(Boolean, default=True)

    paper = relationship("KgPaper", back_populates="tasks")
    task = relationship("KgTask", back_populates="papers")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "task_id"),
    )

    def __repr__(self):
        return f"<KgPaperBelongsToTask(paper={self.paper_id[:8]}, task={self.task_id[:8]})>"


class KgMethodImprovesMethod(Base):
    """方法改进关系 (IMPROVES_UPON)"""
    __tablename__ = "kg_method_improves_method"

    method_a_id = Column(String(32), ForeignKey("kg_methods.method_id", ondelete="CASCADE"), nullable=False)
    method_b_id = Column(String(32), ForeignKey("kg_methods.method_id", ondelete="CASCADE"), nullable=False)
    description = Column(Text, default="")
    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="SET NULL"), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("method_a_id", "method_b_id"),
    )

    def __repr__(self):
        return f"<KgMethodImprovesMethod({self.method_a_id[:8]}→{self.method_b_id[:8]})>"


class KgPaperPublishedIn(Base):
    """论文发表在 venue 关系 (PUBLISHED_IN)"""
    __tablename__ = "kg_paper_published_in"

    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    venue_id = Column(String(32), ForeignKey("kg_venues.venue_id", ondelete="CASCADE"), nullable=False)
    date = Column(String(20), default="")

    __table_args__ = (
        PrimaryKeyConstraint("paper_id", "venue_id"),
    )

    def __repr__(self):
        return f"<KgPaperPublishedIn(paper={self.paper_id[:8]}, venue={self.venue_id[:8]})>"


class KgPaperReportsMetric(Base):
    """论文报告指标关系 (REPORTS_METRIC)"""
    __tablename__ = "kg_paper_reports_metric"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(32), ForeignKey("kg_papers.paper_id", ondelete="CASCADE"), nullable=False)
    metric_id = Column(String(32), ForeignKey("kg_metrics.metric_id", ondelete="CASCADE"), nullable=False)
    value = Column(Float, nullable=True)
    dataset_id = Column(String(32), ForeignKey("kg_datasets.dataset_id", ondelete="SET NULL"), nullable=True)
    condition = Column(String(255), default="")
    notes = Column(Text, default="")

    paper = relationship("KgPaper", back_populates="metrics")
    metric = relationship("KgMetric", back_populates="reports")

    __table_args__ = (
        Index("idx_report_paper_metric", "paper_id", "metric_id", "dataset_id"),
    )

    def __repr__(self):
        return f"<KgPaperReportsMetric(paper={self.paper_id[:8]}, metric={self.metric_id[:8]})>"
