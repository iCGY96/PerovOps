"""
Pydantic schemas for Perovops data structures.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Quantity(BaseModel):
    """Represents a quantity with value and unit."""
    value: float
    unit: str
    uncertainty: Optional[float] = None  # coefficient of variation


class Citation(BaseModel):
    """Citation/provenance information."""
    source: str  # e.g., "paper", "database", "vendor_spec", "heuristic"
    reference: Optional[str] = None  # page, figure, table, or URL
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    timestamp: datetime = Field(default_factory=datetime.now)


class Pedigree(BaseModel):
    """Pedigree matrix for uncertainty quantification."""
    reliability: int = Field(ge=1, le=5, default=3)
    completeness: int = Field(ge=1, le=5, default=3)
    temporal: int = Field(ge=1, le=5, default=3)
    geographical: int = Field(ge=1, le=5, default=3)
    technological: int = Field(ge=1, le=5, default=3)

    def get_uncertainty_factor(self) -> float:
        """Calculate aggregate uncertainty factor from pedigree scores."""
        # Simplified pedigree-to-uncertainty mapping
        scores = [
            self.reliability,
            self.completeness,
            self.temporal,
            self.geographical,
            self.technological,
        ]
        # Lower score = higher uncertainty
        return 1.0 + (sum(scores) - 5) / 20.0


class FlowRef(BaseModel):
    """Reference to an LCI database flow."""
    database: str  # "ecoinvent", "USLCI", "FEDEFL", etc.
    flow_id: Optional[str] = None
    flow_name: str
    unit: str
    is_proxy: bool = False  # True if using a proxy/analogue
    proxy_notes: Optional[str] = None


class MaterialFlow(BaseModel):
    """Material input or output flow."""
    material: str
    quantity: Quantity
    lci_ref: Optional[FlowRef] = None
    density: Optional[Quantity] = None  # for conversions
    recovery_rate: Optional[float] = None  # for solvents
    citations: List[Citation] = Field(default_factory=list)


class EnergyFlow(BaseModel):
    """Energy flow (electricity or thermal)."""
    type: Literal["electricity", "thermal"]
    quantity: Quantity
    region: str = "US-avg"  # eGRID subregion or country
    factor: Optional[float] = None  # gCO2e/kWh
    source_mix: Optional[str] = None  # "grid", "solar", "natural_gas"
    citations: List[Citation] = Field(default_factory=list)


class ProcessStep(BaseModel):
    """A single process step in the device fabrication."""
    id: str
    name: str
    description: Optional[str] = None
    category: Literal[
        "evaporation",
        "sputter",
        "ALD",
        "spin_coat",
        "anneal",
        "lamination",
        "cleaning",
        "screen_print",
        "PECVD",
        "other",
    ]
    inputs: List[MaterialFlow] = Field(default_factory=list)
    outputs: List[MaterialFlow] = Field(default_factory=list)
    energy: List[EnergyFlow] = Field(default_factory=list)
    equipment: Optional[str] = None
    conditions: Dict[str, Any] = Field(default_factory=dict)
    # temp_C, time_min, rate_nm_s, rpm, pressure_pa, etc.
    citations: List[Citation] = Field(default_factory=list)
    uncertainty: Optional[Pedigree] = None
    energy_summary: Optional[str] = None
    carbon_emissions: Optional[str] = None


class Edge(BaseModel):
    """Edge in the process flow graph."""
    source: str  # ProcessStep.id
    target: str  # ProcessStep.id
    flow_type: Literal["material", "energy", "product"]
    label: Optional[str] = None
    description: Optional[str] = None
    energy_input: Optional[str] = None
    carbon_output: Optional[str] = None


class LCIModel(BaseModel):
    """Complete LCI model for a device."""
    name: str
    fu: Quantity  # functional unit, e.g., 1 m^2
    region: str = "US-avg"
    steps: List[ProcessStep] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


# Ingestion schemas


class TableData(BaseModel):
    """Extracted table from PDF."""
    page: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    dataframe: Dict[str, Any]  # JSON-serialized pandas DataFrame


class FigureData(BaseModel):
    """Extracted figure from PDF."""
    page: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    image_base64: Optional[str] = None  # base64-encoded image bytes


class Section(BaseModel):
    """Document section."""
    title: str
    content: str
    page_start: int
    page_end: int


class DocBundle(BaseModel):
    """Complete document parse result."""
    path: str
    sections: List[Section] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    figures: List[FigureData] = Field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# State schema for LangGraph


class PerovopsState(BaseModel):
    """Shared state for LangGraph agents."""
    pdf_path: str
    fu: str = "1 m^2"
    region: str = "US-avg"

    # Agent outputs
    docbundle: Optional[DocBundle] = None
    steps: List[ProcessStep] = Field(default_factory=list)
    normalized: Dict[str, Any] = Field(default_factory=dict)
    links: Dict[str, FlowRef] = Field(default_factory=dict)
    estimates: Dict[str, Any] = Field(default_factory=dict)
    grid_factors: Dict[str, float] = Field(default_factory=dict)
    lci: Optional[LCIModel] = None
    report: Dict[str, Any] = Field(default_factory=dict)

    # Flags
    ingestion_complete: bool = False
    parsing_complete: bool = False
    normalization_complete: bool = False
    linking_complete: bool = False
    estimation_complete: bool = False
    electricity_complete: bool = False
    graph_complete: bool = False

    class Config:
        arbitrary_types_allowed = True
