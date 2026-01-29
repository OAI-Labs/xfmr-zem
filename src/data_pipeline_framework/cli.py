"""
CLI for Data Pipeline Framework
"""

import click
from rich.console import Console
from rich.table import Table
from loguru import logger

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Data Pipeline Framework CLI - ZenML + NemoCurator + DataJuicer"""
    pass


@main.command()
def info():
    """Show framework information"""
    console.print("[bold blue]Data Pipeline Framework[/bold blue]")
    console.print("Version: 0.1.0")
    console.print("\nIntegrations:")
    console.print("  - ZenML: Orchestration & Visualization")
    console.print("  - NemoCurator: Data Curation & Processing")
    console.print("  - DataJuicer: Data Processing Operators")


@main.command()
def operators():
    """List available operators"""
    from data_pipeline_framework.processors import NemoProcessor, DataJuicerProcessor
    
    console.print("\n[bold]NemoCurator Operations:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Operation")
    table.add_column("Description")
    
    for op in NemoProcessor.SUPPORTED_OPERATIONS:
        table.add_row(op, "")
    console.print(table)
    
    console.print("\n[bold]DataJuicer Operators:[/bold]")
    for category, ops in DataJuicerProcessor.OPERATOR_CATEGORIES.items():
        console.print(f"\n[cyan]{category}:[/cyan]")
        for op in ops:
            console.print(f"  - {op}")


@main.command()
@click.option("--domain", type=click.Choice(["legal", "general"]), default="legal")
def templates(domain):
    """Show pipeline templates for domain"""
    if domain == "legal":
        console.print("[bold]Legal Data Pipeline Template[/bold]")
        console.print("""
Steps:
1. unicode_normalization - Fix unicode issues
2. text_cleaning - Remove HTML, clean text
3. remove_header_footer - Remove page headers/footers
4. language_filter - Filter Vietnamese documents
5. length_filter - Remove short documents
6. quality_filter - Filter by quality score
7. exact_dedup - Remove exact duplicates
8. fuzzy_dedup - Remove near-duplicates
        """)
    else:
        console.print("[bold]General Text Pipeline Template[/bold]")
        console.print("""
Steps:
1. unicode_fix - Fix unicode issues
2. clean_html - Remove HTML tags
3. clean_links - Remove URLs
4. normalize_whitespace - Clean whitespace
5. language_filter - Filter by language
6. deduplication - Remove duplicates
        """)


@main.command()
@click.argument("config_file")
@click.option("--input", "-i", required=True, help="Input data path")
@click.option("--output", "-o", required=True, help="Output data path")
def run(config_file, input, output):
    """Run a pipeline from config file"""
    console.print(f"[bold]Running pipeline from:[/bold] {config_file}")
    console.print(f"Input: {input}")
    console.print(f"Output: {output}")
    # TODO: Implement config loading and pipeline execution


if __name__ == "__main__":
    main()
