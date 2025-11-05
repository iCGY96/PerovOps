"""
Web search tools for deep research using ReAct pattern.
Looks up vendor specifications, material properties, and process parameters.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from perovops.utils.config import config

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Web search result."""
    title: str
    url: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    source_type: Literal["vendor", "academic", "database", "other"] = "other"


class ResearchQuery(BaseModel):
    """Research query for material or equipment data."""
    query_type: Literal["density", "deposition_rate", "equipment_power", "chemical_property"]
    material_or_equipment: str
    specific_property: Optional[str] = None
    context: Optional[str] = None


def create_search_client():
    """Create web search client."""
    if config.tavily_api_key and TavilyClient:
        return TavilyClient(api_key=config.tavily_api_key)
    return None


def search_web(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Perform web search using Tavily.

    Args:
        query: Search query
        max_results: Maximum number of results
        include_domains: Specific domains to search

    Returns:
        List of search results
    """
    client = create_search_client()

    if not client:
        logger.warning("Tavily not configured; skipping web search")
        return []

    try:
        logger.info(f"Searching web: {query}")

        response = client.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
        )

        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.5),
                )
            )

        logger.info(f"Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def search_vendor_specs(equipment_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for vendor specifications for equipment.

    Args:
        equipment_name: Name of equipment

    Returns:
        Dictionary with specs if found
    """
    # Try to identify vendor domains
    vendor_domains = [
        "angstromengineering.com",  # Evaporators
        "ajaint.com",  # Sputtering
        "veeco.com",  # ALD
        "laurell.com",  # Spin coaters
    ]

    query = f"{equipment_name} specifications power consumption"

    results = search_web(
        query=query,
        max_results=3,
        include_domains=vendor_domains,
    )

    if not results:
        # Try broader search
        results = search_web(query=query, max_results=5)

    # Extract power specifications from results
    specs = {}

    for result in results:
        content_lower = result.content.lower()

        # Look for power specs
        if "kw" in content_lower or "watt" in content_lower:
            specs["source"] = result.url
            specs["content"] = result.content
            break

    if specs:
        logger.info(f"Found vendor specs for {equipment_name}")
    else:
        logger.info(f"No vendor specs found for {equipment_name}")

    return specs if specs else None


def search_material_density(material_name: str) -> Optional[float]:
    """
    Search for material density.

    Args:
        material_name: Material name

    Returns:
        Density in g/cm^3 if found
    """
    # Academic/database sources
    sources = [
        "pubchem.ncbi.nlm.nih.gov",
        "webbook.nist.gov",
        "matweb.com",
    ]

    query = f"{material_name} density g/cm3"

    results = search_web(
        query=query,
        max_results=5,
        include_domains=sources,
    )

    # Parse density from results
    # This is a simplified implementation
    # In production, would use more sophisticated NLP

    import re

    for result in results:
        # Look for patterns like "density: 5.68 g/cm³"
        matches = re.findall(
            r'density[:\s]+(\d+\.?\d*)\s*g[/\s]cm',
            result.content,
            re.IGNORECASE
        )

        if matches:
            try:
                density = float(matches[0])
                logger.info(f"Found density for {material_name}: {density} g/cm^3")
                return density
            except ValueError:
                continue

    logger.info(f"No density found for {material_name}")
    return None


def search_deposition_rate(
    process_type: str,
    material: str,
) -> Optional[float]:
    """
    Search for typical deposition rates.

    Args:
        process_type: Type of deposition (evaporation, sputtering, ALD)
        material: Material being deposited

    Returns:
        Deposition rate in nm/s if found
    """
    query = f"{material} {process_type} deposition rate nm/s"

    results = search_web(query=query, max_results=5)

    import re

    for result in results:
        # Look for deposition rates
        matches = re.findall(
            r'(\d+\.?\d*)\s*[ÅAa°]?/s|(\d+\.?\d*)\s*nm/s',
            result.content,
            re.IGNORECASE
        )

        if matches:
            for match in matches:
                try:
                    # Convert Å/s to nm/s if needed
                    rate_str = match[0] or match[1]
                    rate = float(rate_str)

                    # If match[0] is populated, it's likely Å/s
                    if match[0] and "å" in result.content[
                        max(0, result.content.lower().find(str(rate))-10):
                        result.content.lower().find(str(rate))+10
                    ].lower():
                        rate = rate / 10.0  # Convert Å/s to nm/s

                    logger.info(
                        f"Found deposition rate for {material} {process_type}: "
                        f"{rate} nm/s"
                    )
                    return rate

                except ValueError:
                    continue

    logger.info(f"No deposition rate found for {material} {process_type}")
    return None


class ReActResearcher:
    """
    ReAct (Reasoning + Acting) agent for deep research.

    Uses iterative reasoning and web search to answer questions
    about materials, equipment, and processes.
    """

    def __init__(self, max_iterations: int = 3):
        """
        Initialize ReAct researcher.

        Args:
            max_iterations: Maximum reasoning iterations
        """
        self.max_iterations = max_iterations
        self.search_history: List[Dict[str, Any]] = []

    def research(self, query: ResearchQuery) -> Optional[Dict[str, Any]]:
        """
        Conduct research using ReAct pattern.

        Args:
            query: Research query

        Returns:
            Research results if found
        """
        logger.info(f"Starting ReAct research: {query.query_type} for {query.material_or_equipment}")

        # Route to specific research function
        if query.query_type == "density":
            density = search_material_density(query.material_or_equipment)
            if density:
                return {
                    "property": "density",
                    "value": density,
                    "unit": "g/cm^3",
                    "material": query.material_or_equipment,
                }

        elif query.query_type == "deposition_rate" and query.context:
            rate = search_deposition_rate(query.context, query.material_or_equipment)
            if rate:
                return {
                    "property": "deposition_rate",
                    "value": rate,
                    "unit": "nm/s",
                    "material": query.material_or_equipment,
                    "process": query.context,
                }

        elif query.query_type == "equipment_power":
            specs = search_vendor_specs(query.material_or_equipment)
            if specs:
                return {
                    "property": "equipment_specs",
                    "equipment": query.material_or_equipment,
                    "data": specs,
                }

        return None

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get history of searches performed."""
        return self.search_history


# Convenience functions for common lookups

def lookup_density(material: str) -> Optional[float]:
    """Quick lookup of material density."""
    return search_material_density(material)


def lookup_deposition_rate(material: str, process: str) -> Optional[float]:
    """Quick lookup of deposition rate."""
    return search_deposition_rate(process, material)


def lookup_equipment_power(equipment: str) -> Optional[Dict[str, Any]]:
    """Quick lookup of equipment specifications."""
    return search_vendor_specs(equipment)