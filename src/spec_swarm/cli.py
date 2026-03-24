"""CLI entry point for SpecSwarm."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="SpecSwarm")
def main():
    """SpecSwarm -- Hardware specification analyzer for embedded software development."""
    pass


@main.command()
@click.option("--port", default=8686, help="Port for SSE transport")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--transport", default="sse", type=click.Choice(["sse", "stdio"]))
def serve(port: int, host: str, transport: str):
    """Start the SpecSwarm MCP server."""
    from .server import create_mcp_server

    mcp = create_mcp_server()

    click.echo(f"SpecSwarm v{__version__} starting on {transport}...")
    if transport == "sse":
        click.echo(f"Listening on http://{host}:{port}/sse")
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


@main.command()
@click.argument("document", type=click.Path(exists=True))
@click.option("--spec-type", "spec_type", default="datasheet",
              type=click.Choice(["datasheet", "reference_manual", "application_note",
                                 "protocol_spec", "requirements", "schematic", "pinout"]))
@click.option("--component", default="", help="Component name (e.g., STM32F407VG)")
def ingest(document: str, spec_type: str, component: str):
    """Ingest a hardware document and extract specifications."""
    from .doc_parser import parse_document
    from .spec_extractor import extract_all
    from .spec_store import SpecStore
    from .models import HardwareSpec, SpecType, Register, PinConfig, ProtocolConfig, TimingConstraint, PowerSpec, MemoryRegion

    click.echo(f"Parsing {document}...")

    try:
        doc = parse_document(document)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"  Format: {doc['format']}, Pages: {doc['pages']}")

    extracted = extract_all(doc["text"], component_name=component)
    stats = extracted.get("extraction_stats", {})

    click.echo(f"\nExtraction results:")
    click.echo(f"  Registers:          {stats.get('registers_found', 0)}")
    click.echo(f"  Pins:               {stats.get('pins_found', 0)}")
    click.echo(f"  Protocols:          {stats.get('protocols_found', 0)}")
    click.echo(f"  Timing constraints: {stats.get('timing_constraints_found', 0)}")
    click.echo(f"  Power specs:        {stats.get('power_specs_found', 0)}")
    click.echo(f"  Memory regions:     {stats.get('memory_regions_found', 0)}")
    click.echo(f"  Constraints:        {stats.get('constraints_found', 0)}")

    # Create a session and store
    store = SpecStore()
    session = store.create_session(str(Path(document).parent.resolve()))

    try:
        st = SpecType(spec_type)
    except ValueError:
        st = SpecType.DATASHEET

    spec = HardwareSpec(
        name=component or extracted.get("name", Path(document).stem),
        category=extracted.get("category", ""),
        source_doc=str(Path(document).resolve()),
        spec_type=st,
        registers=[Register.from_dict(r) for r in extracted.get("registers", [])],
        pins=[PinConfig.from_dict(p) for p in extracted.get("pins", [])],
        protocols=[ProtocolConfig.from_dict(p) for p in extracted.get("protocols", [])],
        timing=[TimingConstraint.from_dict(t) for t in extracted.get("timing", [])],
        power=[PowerSpec.from_dict(p) for p in extracted.get("power", [])],
        memory_map=[MemoryRegion.from_dict(m) for m in extracted.get("memory_map", [])],
        constraints=extracted.get("constraints", []),
    )
    store.add_spec(session.id, spec)

    click.echo(f"\nSession:   {session.id}")
    click.echo(f"Spec ID:   {spec.id}")
    click.echo(f"Component: {spec.name}")
    click.echo(f"Category:  {spec.category}")


@main.command()
def status():
    """Show current spec analysis sessions."""
    from .spec_store import SpecStore

    store = SpecStore()
    sessions = store.list_sessions()

    if not sessions:
        click.echo("No spec sessions found.")
        return

    click.echo(f"Spec sessions ({len(sessions)}):\n")
    for s in sessions:
        click.echo(f"  {s['session_id']}")
        click.echo(f"    Path:     {s['project_path']}")
        click.echo(f"    Created:  {s['created_at'][:19]}")
        click.echo(f"    Specs:    {s['spec_count']}")
        click.echo(f"    Findings: {s['finding_count']}")
        click.echo()


@main.command("list-experts")
def list_experts():
    """List available expert profiles."""
    from .expert_profiler import ExpertProfiler

    profiler = ExpertProfiler()
    profiles = profiler.list_profiles()

    click.echo("Available expert profiles:\n")
    for p in profiles:
        source = Path(p.get("_source_file", "")).stem
        desc = p.get("description", "")
        click.echo(f"  {source:30s} {desc}")


@main.command()
@click.argument("expert_name")
def prompt(expert_name: str):
    """Print the system prompt for an expert profile."""
    from .expert_profiler import ExpertProfiler

    profiler = ExpertProfiler()
    try:
        profile = profiler.load_profile(expert_name)
    except FileNotFoundError:
        click.echo(f"Expert profile not found: {expert_name}", err=True)
        click.echo("Use 'spec-swarm list-experts' to see available profiles.")
        raise SystemExit(1)

    sys_prompt = profile.get("system_prompt", "")
    if not sys_prompt:
        click.echo(f"No system_prompt defined for {expert_name}", err=True)
        raise SystemExit(1)

    click.echo(sys_prompt)


if __name__ == "__main__":
    main()
