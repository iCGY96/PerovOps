"""
Command-line interface for Perovops.
"""

import click
import logging
from pathlib import Path
from perovops.pipeline.supervisor import run_pipeline
from perovops.utils.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Perovops: LCI Graph Builder for Perovskite Device Papers"""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--fu", default="1 m^2", help="Functional unit (default: 1 m^2)")
@click.option("--region", default=None, help="Grid region (default: from config)")
@click.option("--no-ocr", is_flag=True, help="Disable OCR for faster processing")
def build(pdf_path: str, fu: str, region: str, no_ocr: bool):
    """Run ingestion and parsing for a PDF paper."""
    if region is None:
        region = config.default_region

    click.echo(f"Processing: {pdf_path}")
    click.echo(f"FU: {fu}, Region: {region}")

    try:
        result = run_pipeline(
            pdf_path=pdf_path,
            fu=fu,
            region=region,
            use_ocr=not no_ocr,
        )

        docbundle = result.get("docbundle")
        steps = result.get("steps") or []

        click.echo("\nSummary:")
        if docbundle:
            sections_count = len(getattr(docbundle, "sections", []) or [])
            tables_count = len(getattr(docbundle, "tables", []) or [])
            figures_count = len(getattr(docbundle, "figures", []) or [])
            click.echo(f"  Ingested sections={sections_count}, tables={tables_count}, figures={figures_count}")
        else:
            click.echo("  DocBundle not available")

        click.echo(f"  Parsed steps={len(steps)}")
        click.echo("\nDownstream normalization/linking/estimation/electricity/graph stages have been removed;")
        click.echo("no additional exports are produced by this command.")

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        logger.exception("Pipeline failed")
        raise click.Abort()


@cli.command()
@click.option("--key", help="Config key to show")
def config_show(key: str):
    """Show configuration values."""
    if key:
        value = getattr(config, key, None)
        if value:
            click.echo(f"{key}: {value}")
        else:
            click.echo(f"Key '{key}' not found", err=True)
    else:
        click.echo("Configuration:")
        click.echo(f"  Default Region: {config.default_region}")
        click.echo(f"  Default FU: {config.default_fu}")
        click.echo(f"  Solvent Recovery: {config.solvent_recovery_rate}")
        click.echo(f"  Brightway Project: {config.brightway_project}")
        click.echo(f"  Log Level: {config.log_level}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--pattern", default="*.pdf", help="File pattern (default: *.pdf)")
@click.option("--fu", default="1 m^2", help="Functional unit")
@click.option("--region", default=None, help="Grid region")
@click.option("--no-ocr", is_flag=True, help="Disable OCR for faster processing")
def batch(directory: str, pattern: str, fu: str, region: str, no_ocr: bool):
    """Process multiple PDFs in batch mode."""
    import glob

    if region is None:
        region = config.default_region

    pdf_files = glob.glob(str(Path(directory) / pattern))
    click.echo(f"Found {len(pdf_files)} PDF files")

    success_count = 0
    for pdf_path in pdf_files:
        click.echo(f"\nProcessing: {pdf_path}")
        try:
            result = run_pipeline(
                pdf_path=pdf_path,
                fu=fu,
                region=region,
                use_ocr=not no_ocr,
            )
            docbundle = result.get("docbundle")
            steps = result.get("steps") or []
            if docbundle:
                sections = len(getattr(docbundle, "sections", []) or [])
                tables = len(getattr(docbundle, "tables", []) or [])
                figures = len(getattr(docbundle, "figures", []) or [])
                click.echo(f"  Ingested sections={sections}, tables={tables}, figures={figures}")
            else:
                click.echo("  DocBundle not available")
            click.echo(f"  Parsed steps={len(steps)}")
            success_count += 1
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)

    click.echo(f"\nBatch complete: {success_count}/{len(pdf_files)} succeeded")


def main():
    """Entry point for CLI."""
    cli()
