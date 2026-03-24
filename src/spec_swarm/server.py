"""MCP Server for SpecSwarm -- hardware specification analysis tools."""

import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    HardwareSpec, SpecSession, SpecType, Register, PinConfig,
    ProtocolConfig, TimingConstraint, PowerSpec, MemoryRegion, now_iso,
)
from .spec_store import SpecStore
from .expert_profiler import ExpertProfiler
from .session_manager import SpecSessionManager, SpecVerification

_log = logging.getLogger(__name__)


@dataclass
class AppContext:
    store: SpecStore
    profiler: ExpertProfiler
    verification_mgr: SpecSessionManager


def create_app_context() -> AppContext:
    store = SpecStore()
    profiler = ExpertProfiler()
    verification_mgr = SpecSessionManager()
    return AppContext(store=store, profiler=profiler, verification_mgr=verification_mgr)


def create_mcp_server():
    """Create and configure the MCP server with all spec analysis tools."""
    from mcp.server.fastmcp import FastMCP, Context
    from contextlib import asynccontextmanager
    from collections.abc import AsyncIterator

    def _get_app(ctx: Optional[Context]) -> AppContext:
        """Extract AppContext from MCP Context."""
        assert ctx is not None, "MCP Context not injected by FastMCP"
        return ctx.request_context.lifespan_context

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        ctx = create_app_context()
        yield ctx

    mcp = FastMCP("SpecSwarm", lifespan=lifespan)

    # ── Session Management ───────────────────────────────────────────

    @mcp.tool(
        name="spec_start_session",
        description="Start a spec analysis session for a project. Returns session_id for use with other tools.",
    )
    def _spec_start_session(
        project_path: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        session = app.store.create_session(project_path)
        return json.dumps({
            "session_id": session.id,
            "project_path": session.project_path,
            "created_at": session.created_at,
            "status": "active",
        }, indent=2)

    @mcp.tool(
        name="spec_list_sessions",
        description="List all spec analysis sessions.",
    )
    def _spec_list_sessions(
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        sessions = app.store.list_sessions()
        if not sessions:
            return json.dumps({"sessions": [], "message": "No sessions found."})
        return json.dumps({"sessions": sessions}, indent=2)

    # ── Document Ingestion ───────────────────────────────────────────

    @mcp.tool(
        name="spec_ingest",
        description=(
            "Ingest a document (PDF, text, markdown) and extract hardware specifications. "
            "Supports datasheets, reference manuals, application notes. "
            "Returns extracted registers, pins, protocols, timing constraints, power specs, "
            "and memory map regions."
        ),
    )
    def _spec_ingest(
        session_id: str,
        document_path: str,
        spec_type: str = "datasheet",
        component_name: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        session = app.store.get_session(session_id)

        # Validate document_path exists and is a file
        from pathlib import Path as _Path
        doc_path = _Path(document_path)
        if not doc_path.is_file():
            return json.dumps({"error": f"File not found: {document_path}"})

        # Parse document
        from .doc_parser import parse_document
        try:
            doc = parse_document(document_path)
        except ImportError as e:
            return json.dumps({"error": str(e)})
        except FileNotFoundError as e:
            return json.dumps({"error": str(e)})

        # Extract structured data
        from .spec_extractor import extract_all
        extracted = extract_all(doc["text"], component_name=component_name)

        # Determine spec type
        try:
            st = SpecType(spec_type)
        except ValueError:
            st = SpecType.DATASHEET

        # Build HardwareSpec
        spec = HardwareSpec(
            name=component_name or extracted.get("name", ""),
            category=extracted.get("category", ""),
            source_doc=document_path,
            spec_type=st,
            registers=[Register.from_dict(r) for r in extracted.get("registers", [])],
            pins=[PinConfig.from_dict(p) for p in extracted.get("pins", [])],
            protocols=[ProtocolConfig.from_dict(p) for p in extracted.get("protocols", [])],
            timing=[TimingConstraint.from_dict(t) for t in extracted.get("timing", [])],
            power=[PowerSpec.from_dict(p) for p in extracted.get("power", [])],
            memory_map=[MemoryRegion.from_dict(m) for m in extracted.get("memory_map", [])],
            constraints=extracted.get("constraints", []),
        )

        app.store.add_spec(session_id, spec)

        result = {
            "spec_id": spec.id,
            "name": spec.name,
            "category": spec.category,
            "source": document_path,
            "format": doc["format"],
            "pages": doc["pages"],
            "extraction_stats": extracted.get("extraction_stats", {}),
        }
        return json.dumps(result, indent=2)

    @mcp.tool(
        name="spec_add_manual",
        description=(
            "Manually add a hardware specification (register, pin, protocol, timing constraint). "
            "Provide spec_json as a JSON string with fields matching HardwareSpec: "
            "name, category, registers, pins, protocols, timing, power, memory_map, constraints, notes, tags."
        ),
    )
    def _spec_add_manual(
        session_id: str,
        spec_json: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        app.store.get_session(session_id)  # validate session exists

        try:
            data = json.loads(spec_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        spec = HardwareSpec.from_dict(data)
        app.store.add_spec(session_id, spec)

        return json.dumps({
            "spec_id": spec.id,
            "name": spec.name,
            "category": spec.category,
            "status": "added",
            "registers": len(spec.registers),
            "pins": len(spec.pins),
            "protocols": len(spec.protocols),
            "timing": len(spec.timing),
            "power": len(spec.power),
            "memory_regions": len(spec.memory_map),
        }, indent=2)

    # ── Query Tools ──────────────────────────────────────────────────

    @mcp.tool(
        name="spec_get_registers",
        description=(
            "Get register map for a component. Optionally filter by component name, "
            "peripheral keyword, or address range (hex)."
        ),
    )
    def _spec_get_registers(
        session_id: str,
        component_name: str = "",
        peripheral_filter: str = "",
        address_start: str = "",
        address_end: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for reg in spec.registers:
                # Filter by peripheral keyword
                if peripheral_filter:
                    pf = peripheral_filter.upper()
                    if pf not in reg.name.upper() and pf not in reg.description.upper():
                        continue

                # Filter by address range
                if address_start:
                    try:
                        if int(reg.address, 16) < int(address_start, 16):
                            continue
                    except ValueError:
                        pass
                if address_end:
                    try:
                        if int(reg.address, 16) > int(address_end, 16):
                            continue
                    except ValueError:
                        pass

                entry = reg.to_dict()
                entry["component"] = spec.name
                results.append(entry)

        return json.dumps({
            "registers": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_get_pins",
        description="Get pin configuration for a component. Optionally filter by component name or function keyword.",
    )
    def _spec_get_pins(
        session_id: str,
        component_name: str = "",
        function_filter: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for pin in spec.pins:
                if function_filter:
                    ff = function_filter.upper()
                    if ff not in pin.function.upper() and ff not in pin.pin.upper():
                        continue
                entry = pin.to_dict()
                entry["component"] = spec.name
                results.append(entry)

        return json.dumps({
            "pins": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_get_protocols",
        description="Get communication protocol configurations. Optionally filter by protocol type (SPI, I2C, UART, CAN, etc.).",
    )
    def _spec_get_protocols(
        session_id: str,
        protocol_filter: str = "",
        component_name: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for proto in spec.protocols:
                if protocol_filter:
                    pf = protocol_filter.upper()
                    if pf not in proto.protocol.upper() and pf not in proto.instance.upper():
                        continue
                entry = proto.to_dict()
                entry["component"] = spec.name
                results.append(entry)

        return json.dumps({
            "protocols": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_get_timing",
        description="Get timing constraints. Optionally filter by critical-only or parameter keyword.",
    )
    def _spec_get_timing(
        session_id: str,
        critical_only: bool = False,
        parameter_filter: str = "",
        component_name: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for timing in spec.timing:
                if critical_only and not timing.critical:
                    continue
                if parameter_filter:
                    if parameter_filter.lower() not in timing.parameter.lower():
                        continue
                entry = timing.to_dict()
                entry["component"] = spec.name
                results.append(entry)

        return json.dumps({
            "timing_constraints": results,
            "count": len(results),
            "critical_count": sum(1 for t in results if t.get("critical", False)),
        }, indent=2)

    @mcp.tool(
        name="spec_get_memory_map",
        description="Get memory map regions. Optionally filter by component name or region type.",
    )
    def _spec_get_memory_map(
        session_id: str,
        component_name: str = "",
        region_filter: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for region in spec.memory_map:
                if region_filter:
                    rf = region_filter.upper()
                    if rf not in region.name.upper() and rf not in region.description.upper():
                        continue
                entry = region.to_dict()
                entry["component"] = spec.name
                results.append(entry)

        return json.dumps({
            "memory_regions": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_get_constraints",
        description="Get all hardware constraints that software must respect. Returns free-form constraints from datasheets.",
    )
    def _spec_get_constraints(
        session_id: str,
        component_name: str = "",
        keyword: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        results: list[dict] = []
        for spec in specs:
            if component_name and component_name.lower() not in spec.name.lower():
                continue

            for constraint in spec.constraints:
                if keyword and keyword.lower() not in constraint.lower():
                    continue
                results.append({
                    "component": spec.name,
                    "constraint": constraint,
                })

        return json.dumps({
            "constraints": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_search",
        description="Search specs by keyword across all components and fields. Returns matching specs with context.",
    )
    def _spec_search(
        session_id: str,
        query: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)
        query_lower = query.lower()

        results: list[dict] = []
        for spec in specs:
            matches: list[str] = []

            # Search in name, category, part_number, manufacturer
            for field_name in ("name", "category", "part_number", "manufacturer"):
                val = getattr(spec, field_name, "")
                if val and query_lower in val.lower():
                    matches.append(f"{field_name}: {val}")

            # Search in registers
            for reg in spec.registers:
                if query_lower in reg.name.lower() or query_lower in reg.description.lower():
                    matches.append(f"register: {reg.name} ({reg.address})")
                for fld in reg.fields:
                    if query_lower in fld.get("name", "").lower():
                        matches.append(f"register field: {reg.name}.{fld['name']}")

            # Search in pins
            for pin in spec.pins:
                if query_lower in pin.pin.lower() or query_lower in pin.function.lower():
                    matches.append(f"pin: {pin.pin} / {pin.function}")

            # Search in protocols
            for proto in spec.protocols:
                if query_lower in proto.protocol.lower() or query_lower in proto.instance.lower():
                    matches.append(f"protocol: {proto.instance} ({proto.protocol})")

            # Search in timing
            for timing in spec.timing:
                if query_lower in timing.parameter.lower():
                    matches.append(f"timing: {timing.parameter}")

            # Search in power
            for pwr in spec.power:
                if query_lower in pwr.rail.lower() or query_lower in pwr.notes.lower():
                    matches.append(f"power: {pwr.rail}")

            # Search in memory map
            for region in spec.memory_map:
                if query_lower in region.name.lower() or query_lower in region.description.lower():
                    matches.append(f"memory: {region.name} ({region.start_address})")

            # Search in constraints
            for constraint in spec.constraints:
                if query_lower in constraint.lower():
                    matches.append(f"constraint: {constraint[:80]}...")

            # Search in notes and tags
            for note in spec.notes:
                if query_lower in note.lower():
                    matches.append(f"note: {note[:80]}...")
            for tag in spec.tags:
                if query_lower in tag.lower():
                    matches.append(f"tag: {tag}")

            if matches:
                results.append({
                    "spec_id": spec.id,
                    "component": spec.name,
                    "category": spec.category,
                    "matches": matches,
                    "match_count": len(matches),
                })

        results.sort(key=lambda r: r["match_count"], reverse=True)
        return json.dumps({
            "query": query,
            "results": results,
            "total_matches": sum(r["match_count"] for r in results),
        }, indent=2)

    # ── Analysis Tools ───────────────────────────────────────────────

    @mcp.tool(
        name="spec_check_conflicts",
        description=(
            "Check for conflicts between components: pin collisions (same pin used by "
            "multiple peripherals), bus conflicts (address collisions on I2C), "
            "power budget violations (total current exceeds supply)."
        ),
    )
    def _spec_check_conflicts(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        conflicts: list[dict] = []

        # 1. Pin collision detection
        pin_usage: dict[str, list[dict]] = {}  # pin_name -> [{"component", "function"}]
        for spec in specs:
            for pin in spec.pins:
                key = pin.pin.upper()
                if not key:
                    continue
                if key not in pin_usage:
                    pin_usage[key] = []
                pin_usage[key].append({
                    "component": spec.name,
                    "function": pin.function,
                    "direction": pin.direction,
                })

        for pin_name, usages in pin_usage.items():
            if len(usages) > 1:
                # Check if it's actually a conflict (same pin, different functions)
                functions = set(u["function"] for u in usages)
                if len(functions) > 1:
                    conflicts.append({
                        "type": "pin_collision",
                        "severity": "high",
                        "pin": pin_name,
                        "usages": usages,
                        "message": (
                            f"Pin {pin_name} is used by multiple peripherals: "
                            + ", ".join(f"{u['component']}/{u['function']}" for u in usages)
                        ),
                    })

        # 2. I2C address collision
        i2c_addresses: dict[str, list[dict]] = {}  # address -> [{"component", "bus"}]
        for spec in specs:
            for proto in spec.protocols:
                if proto.protocol.upper() == "I2C" and proto.notes:
                    # Extract address from notes
                    addr_match = re.search(r"0x[0-9A-Fa-f]{2}", proto.notes)
                    if addr_match:
                        addr = addr_match.group(0).upper()
                        bus = proto.instance
                        if addr not in i2c_addresses:
                            i2c_addresses[addr] = []
                        i2c_addresses[addr].append({
                            "component": spec.name,
                            "bus": bus,
                        })

        for addr, devices in i2c_addresses.items():
            if len(devices) > 1:
                # Check if they're on the same bus
                buses = set(d["bus"] for d in devices)
                for bus in buses:
                    bus_devices = [d for d in devices if d["bus"] == bus]
                    if len(bus_devices) > 1:
                        conflicts.append({
                            "type": "i2c_address_collision",
                            "severity": "high",
                            "address": addr,
                            "bus": bus,
                            "devices": bus_devices,
                            "message": (
                                f"I2C address {addr} on {bus} used by: "
                                + ", ".join(d["component"] for d in bus_devices)
                            ),
                        })

        # 3. Power budget check
        total_current_ma = 0.0
        supply_current_ma = 0.0
        current_consumers: list[dict] = []
        for spec in specs:
            for pwr in spec.power:
                current_str = pwr.max_current
                if current_str:
                    current_match = re.search(r"(\d+(?:\.\d+)?)\s*(mA|uA|A)", current_str)
                    if current_match:
                        val = float(current_match.group(1))
                        unit = current_match.group(2)
                        if unit == "uA":
                            val /= 1000.0
                        elif unit == "A":
                            val *= 1000.0
                        # Check if this looks like a supply spec or consumer spec
                        if spec.category in ("mcu", "power"):
                            supply_current_ma = max(supply_current_ma, val)
                        else:
                            total_current_ma += val
                            current_consumers.append({
                                "component": spec.name,
                                "rail": pwr.rail,
                                "current_ma": val,
                            })

        if supply_current_ma > 0 and total_current_ma > supply_current_ma:
            conflicts.append({
                "type": "power_budget_violation",
                "severity": "critical",
                "total_demand_ma": total_current_ma,
                "supply_capacity_ma": supply_current_ma,
                "consumers": current_consumers,
                "message": (
                    f"Total current demand ({total_current_ma:.1f} mA) exceeds "
                    f"supply capacity ({supply_current_ma:.1f} mA)"
                ),
            })

        # 4. Memory overlap detection
        all_regions: list[dict] = []
        for spec in specs:
            for region in spec.memory_map:
                if region.start_address:
                    try:
                        start = int(region.start_address, 16)
                        end = int(region.end_address, 16) if region.end_address else start
                        all_regions.append({
                            "component": spec.name,
                            "name": region.name,
                            "start": start,
                            "end": end,
                        })
                    except ValueError:
                        continue

        for i, r1 in enumerate(all_regions):
            for r2 in all_regions[i + 1:]:
                if r1["start"] <= r2["end"] and r2["start"] <= r1["end"]:
                    # Overlap from different components (same component overlaps are expected)
                    if r1["component"] != r2["component"]:
                        conflicts.append({
                            "type": "memory_overlap",
                            "severity": "high",
                            "region_a": f"{r1['component']}/{r1['name']} (0x{r1['start']:08X}-0x{r1['end']:08X})",
                            "region_b": f"{r2['component']}/{r2['name']} (0x{r2['start']:08X}-0x{r2['end']:08X})",
                            "message": f"Memory regions overlap: {r1['name']} and {r2['name']}",
                        })

        # Store findings
        for conflict in conflicts:
            app.store.add_finding(session_id, {
                "type": "conflict",
                "conflict": conflict,
            })

        return json.dumps({
            "conflicts": conflicts,
            "total_conflicts": len(conflicts),
            "by_severity": {
                "critical": sum(1 for c in conflicts if c.get("severity") == "critical"),
                "high": sum(1 for c in conflicts if c.get("severity") == "high"),
                "medium": sum(1 for c in conflicts if c.get("severity") == "medium"),
            },
        }, indent=2)

    @mcp.tool(
        name="spec_suggest_experts",
        description="Suggest spec expert profiles based on components and protocols used in the session.",
    )
    def _spec_suggest_experts(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        spec_dicts = [s.to_dict() for s in specs]
        protocols_used = []
        categories_used = []
        for s in specs:
            for p in s.protocols:
                if p.protocol:
                    protocols_used.append(p.protocol)
            if s.category:
                categories_used.append(s.category)

        suggestions = app.profiler.suggest_experts(
            spec_dicts,
            protocols_used=protocols_used,
            categories_used=categories_used,
        )

        return json.dumps({
            "suggestions": suggestions,
            "count": len(suggestions),
        }, indent=2)

    @mcp.tool(
        name="spec_export_for_arch",
        description=(
            "Export specs as architectural constraints for ArchSwarm. "
            "Converts timing constraints, pin configs, and protocol requirements "
            "into architecture-level constraints. Posts findings to swarm-kb."
        ),
    )
    def _spec_export_for_arch(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        specs = app.store.get_specs(session_id)

        arch_constraints: list[dict] = []

        for spec in specs:
            component = spec.name

            # Convert timing constraints to architectural constraints
            for timing in spec.timing:
                constraint_text = ""
                if timing.max_value:
                    constraint_text = (
                        f"{timing.parameter}: must not exceed {timing.max_value}"
                    )
                elif timing.min_value:
                    constraint_text = (
                        f"{timing.parameter}: must be at least {timing.min_value}"
                    )
                elif timing.typ_value:
                    constraint_text = (
                        f"{timing.parameter}: typical {timing.typ_value} (budget accordingly)"
                    )

                if constraint_text:
                    if timing.condition:
                        constraint_text += f" (condition: {timing.condition})"

                    arch_constraints.append({
                        "source": "timing",
                        "component": component,
                        "constraint": constraint_text,
                        "critical": timing.critical,
                        "category": "hw-timing",
                    })

            # Convert pin configs to architectural constraints
            used_pins: set[str] = set()
            for pin in spec.pins:
                if pin.pin and pin.function:
                    used_pins.add(pin.pin)
                    arch_constraints.append({
                        "source": "pin_assignment",
                        "component": component,
                        "constraint": f"Pin {pin.pin} = {pin.function} -- cannot be used for other purposes",
                        "critical": False,
                        "category": "hw-pin",
                    })

            # Convert protocol configs to architectural constraints
            for proto in spec.protocols:
                constraint_parts = [f"{proto.instance} ({proto.protocol})"]
                if proto.speed:
                    constraint_parts.append(f"max speed: {proto.speed}")
                if proto.mode:
                    constraint_parts.append(f"mode: {proto.mode}")
                if proto.role:
                    constraint_parts.append(f"role: {proto.role}")

                arch_constraints.append({
                    "source": "protocol",
                    "component": component,
                    "constraint": f"Protocol {' -- '.join(constraint_parts)}",
                    "critical": False,
                    "category": "hw-protocol",
                })

            # Convert power specs to architectural constraints
            for pwr in spec.power:
                parts = [f"Power rail {pwr.rail}"]
                if pwr.min_voltage and pwr.max_voltage:
                    parts.append(f"voltage range: {pwr.min_voltage} to {pwr.max_voltage}")
                if pwr.max_current:
                    parts.append(f"max current: {pwr.max_current}")

                arch_constraints.append({
                    "source": "power",
                    "component": component,
                    "constraint": " -- ".join(parts),
                    "critical": True,
                    "category": "hw-power",
                })

            # Convert memory map to architectural constraints
            for region in spec.memory_map:
                parts = [f"Memory region {region.name}"]
                if region.start_address:
                    parts.append(f"starts at {region.start_address}")
                if region.size:
                    parts.append(f"size: {region.size}")
                if region.access:
                    parts.append(f"access: {region.access}")

                arch_constraints.append({
                    "source": "memory",
                    "component": component,
                    "constraint": " -- ".join(parts),
                    "critical": False,
                    "category": "hw-memory",
                })

            # Pass through free-form constraints
            for constraint_text in spec.constraints:
                arch_constraints.append({
                    "source": "datasheet",
                    "component": component,
                    "constraint": constraint_text,
                    "critical": True,
                    "category": "hw-constraint",
                })

        # Post to swarm-kb
        kb_posted = app.store.post_to_swarm_kb(
            session_id,
            tool="spec",
            category="hw-constraint",
            data={
                "session_id": session_id,
                "constraints": arch_constraints,
                "exported_at": now_iso(),
            },
        )

        # Also store as findings
        for ac in arch_constraints:
            app.store.add_finding(session_id, {
                "type": "arch_export",
                "constraint": ac,
            })

        return json.dumps({
            "arch_constraints": arch_constraints,
            "total_constraints": len(arch_constraints),
            "by_category": {
                cat: sum(1 for c in arch_constraints if c.get("category") == cat)
                for cat in set(c.get("category", "") for c in arch_constraints)
            },
            "posted_to_swarm_kb": kb_posted,
        }, indent=2)

    # ── Report Generation ────────────────────────────────────────────

    @mcp.tool(
        name="spec_generate_report",
        description=(
            "Generate a Hardware Specification Report from the current session. "
            "This report serves as input for ArchSwarm architecture analysis. "
            "If verification_session is provided, the report includes a Verification "
            "Status section with confirmed, corrected, and disputed items. "
            "Saved to swarm-kb and returned as markdown."
        ),
    )
    def _spec_generate_report(
        session_id: str,
        output_path: str = "",
        verification_session: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        from pathlib import Path
        from .report_generator import generate_report

        app = _get_app(ctx)
        session = app.store.get_session(session_id)
        specs = session.specs

        if not specs:
            return json.dumps({
                "error": "No specs in session. Ingest or add specs first.",
            })

        # Derive project name from session project_path
        project_name = ""
        if session.project_path:
            project_name = Path(session.project_path).name
        if not project_name:
            project_name = session_id

        # Get verification summary if a verification session was provided
        verification_summary = None
        if verification_session:
            try:
                verification_summary = app.verification_mgr.get_summary(verification_session)
            except KeyError:
                _log.warning("Verification session %s not found, skipping verification section", verification_session)

        report_md = generate_report(
            specs, session_id, project_name=project_name,
            verification_summary=verification_summary,
        )

        # Save report to session directory
        sess_dir = app.store._session_dir(session_id)
        report_file = sess_dir / "spec_report.md"
        report_file.write_text(report_md, encoding="utf-8")

        # Optionally write to output_path
        if output_path:
            out = Path(output_path).resolve()
            cwd = Path.cwd().resolve()
            if not out.is_relative_to(cwd):
                return json.dumps({"error": "output_path must be under current directory"})
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(report_md, encoding="utf-8")
            except OSError:
                pass

        # Post to swarm-kb as a finding with category="spec-report"
        kb_posted = app.store.post_to_swarm_kb(
            session_id,
            tool="spec",
            category="spec-report",
            data={
                "session_id": session_id,
                "project_name": project_name,
                "report": report_md,
                "exported_at": now_iso(),
            },
        )

        # Store as finding
        app.store.add_finding(session_id, {
            "type": "spec_report",
            "report_path": str(report_file),
            "kb_posted": kb_posted,
            "verification_session": verification_session or None,
        })

        # Build summary stats
        total_registers = sum(len(s.registers) for s in specs)
        total_pins = sum(len(s.pins) for s in specs)
        total_protocols = sum(len(s.protocols) for s in specs)
        total_timing = sum(len(s.timing) for s in specs)
        total_power = sum(len(s.power) for s in specs)
        total_memory = sum(len(s.memory_map) for s in specs)

        from .report_generator import extract_arch_constraints
        arch_constraints = extract_arch_constraints(specs)

        result = {
            "report": report_md,
            "report_path": str(report_file),
            "posted_to_swarm_kb": kb_posted,
            "stats": {
                "components": len(specs),
                "registers": total_registers,
                "pins": total_pins,
                "protocols": total_protocols,
                "timing_constraints": total_timing,
                "power_specs": total_power,
                "memory_regions": total_memory,
                "arch_constraints_derived": len(arch_constraints),
            },
        }

        if verification_summary is not None:
            result["verification"] = {
                "session": verification_session,
                "total_verifications": verification_summary.get("total_verifications", 0),
                "confirmed": verification_summary.get("confirmed", 0),
                "disputed": verification_summary.get("disputed", 0),
                "corrected": verification_summary.get("corrected", 0),
                "confirmation_rate": verification_summary.get("confirmation_rate", 0.0),
            }

        return json.dumps(result, indent=2)

    # ── Summary ──────────────────────────────────────────────────────

    @mcp.tool(
        name="spec_get_summary",
        description=(
            "Get summary of all specs in a session: components, total registers, "
            "pins allocated, protocols configured, timing constraints, power specs, "
            "memory regions, and findings."
        ),
    )
    def _spec_get_summary(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        session = app.store.get_session(session_id)
        specs = session.specs

        components: list[dict] = []
        total_registers = 0
        total_pins = 0
        total_protocols = 0
        total_timing = 0
        total_power = 0
        total_memory = 0

        for spec in specs:
            reg_count = len(spec.registers)
            pin_count = len(spec.pins)
            proto_count = len(spec.protocols)
            timing_count = len(spec.timing)
            power_count = len(spec.power)
            mem_count = len(spec.memory_map)

            total_registers += reg_count
            total_pins += pin_count
            total_protocols += proto_count
            total_timing += timing_count
            total_power += power_count
            total_memory += mem_count

            components.append({
                "spec_id": spec.id,
                "name": spec.name,
                "category": spec.category,
                "spec_type": spec.spec_type.value if isinstance(spec.spec_type, SpecType) else spec.spec_type,
                "source_doc": spec.source_doc,
                "registers": reg_count,
                "pins": pin_count,
                "protocols": proto_count,
                "timing_constraints": timing_count,
                "power_specs": power_count,
                "memory_regions": mem_count,
                "constraints": len(spec.constraints),
            })

        critical_timing = sum(
            1 for spec in specs
            for t in spec.timing if t.critical
        )

        return json.dumps({
            "session_id": session_id,
            "project_path": session.project_path,
            "created_at": session.created_at,
            "components": components,
            "totals": {
                "components": len(specs),
                "registers": total_registers,
                "pins_allocated": total_pins,
                "protocols_configured": total_protocols,
                "timing_constraints": total_timing,
                "critical_timing": critical_timing,
                "power_specs": total_power,
                "memory_regions": total_memory,
                "findings": len(session.findings),
            },
        }, indent=2)

    # ── Verification Session Management ─────────────────────────────

    @mcp.tool(
        name="spec_start_verification",
        description=(
            "Start a multi-agent verification session for extracted specs. "
            "Experts will cross-check all data."
        ),
    )
    def _spec_start_verification(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        # Validate the spec session exists
        spec_session = app.store.get_session(session_id)
        specs = spec_session.specs
        if not specs:
            return json.dumps({"error": "No specs in session. Ingest or add specs first."})

        vsess_id = app.verification_mgr.start_session(
            project_path=spec_session.project_path,
            name=f"verification-{session_id}",
        )

        components = []
        for s in specs:
            components.append({
                "spec_id": s.id,
                "name": s.name or s.id,
                "registers": len(s.registers),
                "pins": len(s.pins),
                "protocols": len(s.protocols),
                "timing": len(s.timing),
            })

        return json.dumps({
            "verification_session": vsess_id,
            "spec_session": session_id,
            "components_to_verify": components,
            "phases": [
                {"phase": 1, "name": "Independent Verification"},
                {"phase": 2, "name": "Cross-Check"},
                {"phase": 3, "name": "Resolve Disputes"},
                {"phase": 4, "name": "Generate Report"},
            ],
            "status": "active",
        }, indent=2)

    # ── Expert Coordination ───────────────────────────────────────────

    @mcp.tool(
        name="spec_claim_component",
        description="Claim a component/spec for verification. Prevents duplicate work.",
    )
    def _spec_claim_component(
        session_id: str,
        spec_id: str,
        expert_role: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        result = app.verification_mgr.claim_spec(session_id, spec_id, expert_role)
        return json.dumps(result, indent=2)

    @mcp.tool(
        name="spec_release_component",
        description="Release a claimed component.",
    )
    def _spec_release_component(
        session_id: str,
        spec_id: str,
        expert_role: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        result = app.verification_mgr.release_spec(session_id, spec_id, expert_role)
        return json.dumps(result, indent=2)

    # ── Verification ──────────────────────────────────────────────────

    @mcp.tool(
        name="spec_verify",
        description=(
            "Verify an extracted spec field. Confirm accuracy, dispute if wrong, "
            "or correct with evidence."
        ),
    )
    def _spec_verify(
        session_id: str,
        spec_id: str,
        expert_role: str,
        field_path: str,
        status: str,
        evidence: str = "",
        corrected_value: str = "",
        original_value: str = "",
        confidence: float = 0.9,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)

        if status not in ("confirm", "dispute", "correct"):
            return json.dumps({"error": f"Invalid status: {status}. Must be confirm|dispute|correct."})
        if status == "correct" and not corrected_value:
            return json.dumps({"error": "corrected_value is required when status='correct'."})
        if not evidence:
            return json.dumps({"error": "evidence is required. Include page number and table/figure reference."})

        verification = SpecVerification(
            spec_id=spec_id,
            field_path=field_path,
            expert_role=expert_role,
            status=status,
            original_value=original_value,
            corrected_value=corrected_value,
            evidence=evidence,
            confidence=confidence,
        )
        vid = app.verification_mgr.post_verification(session_id, verification)

        return json.dumps({
            "verification_id": vid,
            "spec_id": spec_id,
            "field_path": field_path,
            "status": status,
            "expert_role": expert_role,
            "evidence": evidence,
            "recorded": True,
        }, indent=2)

    @mcp.tool(
        name="spec_get_verifications",
        description="Get verification results. Filter by spec, expert, or status.",
    )
    def _spec_get_verifications(
        session_id: str,
        spec_id: str = "",
        expert_role: str = "",
        status: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        results = app.verification_mgr.get_verifications(
            session_id, spec_id=spec_id, expert_role=expert_role, status=status,
        )
        return json.dumps({
            "verifications": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool(
        name="spec_verification_status",
        description=(
            "Get aggregated verification status for a component: "
            "how many confirmed, disputed, corrected."
        ),
    )
    def _spec_verification_status(
        session_id: str,
        spec_id: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        if spec_id:
            result = app.verification_mgr.get_verification_status(session_id, spec_id)
            return json.dumps(result, indent=2)

        # If no spec_id, return status for all specs in the verification session
        verifications = app.verification_mgr.get_verifications(session_id)
        spec_ids = sorted({v["spec_id"] for v in verifications})
        all_status = []
        for sid in spec_ids:
            all_status.append(
                app.verification_mgr.get_verification_status(session_id, sid)
            )

        total_checks = sum(s["total_checks"] for s in all_status)
        total_confirmed = sum(s["confirmed"] for s in all_status)
        total_disputed = sum(s["disputed"] for s in all_status)
        total_corrected = sum(s["corrected"] for s in all_status)

        return json.dumps({
            "session_id": session_id,
            "specs": all_status,
            "totals": {
                "total_checks": total_checks,
                "confirmed": total_confirmed,
                "disputed": total_disputed,
                "corrected": total_corrected,
                "confirmation_rate": round(total_confirmed / total_checks * 100, 1) if total_checks > 0 else 0.0,
            },
        }, indent=2)

    # ── Messaging ─────────────────────────────────────────────────────

    @mcp.tool(
        name="spec_send_message",
        description="Send a message to another spec expert about a verification question.",
    )
    def _spec_send_message(
        session_id: str,
        sender: str,
        recipient: str,
        content: str,
        context_id: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        msg_id = app.verification_mgr.send_message(
            session_id, sender, recipient, content, context_id=context_id,
        )
        return json.dumps({"message_id": msg_id, "sent": True}, indent=2)

    @mcp.tool(
        name="spec_get_inbox",
        description="Get pending messages for a spec expert.",
    )
    def _spec_get_inbox(
        session_id: str,
        expert_role: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        messages = app.verification_mgr.get_inbox(session_id, expert_role)
        return json.dumps({
            "messages": messages,
            "count": len(messages),
        }, indent=2)

    @mcp.tool(
        name="spec_broadcast",
        description="Broadcast a message to all spec experts.",
    )
    def _spec_broadcast(
        session_id: str,
        sender: str,
        content: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        msg_id = app.verification_mgr.broadcast(session_id, sender, content)
        return json.dumps({"message_id": msg_id, "broadcast": True}, indent=2)

    # ── Phases ────────────────────────────────────────────────────────

    @mcp.tool(
        name="spec_mark_phase_done",
        description="Mark that an expert completed a verification phase.",
    )
    def _spec_mark_phase_done(
        session_id: str,
        expert_role: str,
        phase: int,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        result = app.verification_mgr.mark_phase_done(session_id, expert_role, phase)
        return json.dumps(result, indent=2)

    @mcp.tool(
        name="spec_check_phase_ready",
        description="Check if all experts completed a verification phase.",
    )
    def _spec_check_phase_ready(
        session_id: str,
        phase: int,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        result = app.verification_mgr.check_phase_ready(session_id, phase)
        return json.dumps(result, indent=2)

    # ── Debate Trigger ────────────────────────────────────────────────

    @mcp.tool(
        name="spec_start_debate",
        description=(
            "Start a debate when experts disagree on spec interpretation. "
            "Uses swarm-kb debate engine."
        ),
    )
    def _spec_start_debate(
        session_id: str,
        topic: str,
        context: str = "",
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)

        debate_id: str
        try:
            from swarm_kb.debate_engine import DebateEngine
            from pathlib import Path
            debates_path = Path.home() / ".swarm-kb" / "debates" / "active"
            debates_path.mkdir(parents=True, exist_ok=True)
            engine = DebateEngine(debates_path)
            debate = engine.start_debate(
                topic=topic,
                context=context,
                source_tool="spec",
            )
            debate_id = debate.id
        except ImportError:
            _log.warning("swarm-kb debate engine not available; generating local debate id")
            debate_id = "dbt-" + secrets.token_hex(4)
        except Exception as exc:
            _log.warning("Failed to start swarm-kb debate: %s", exc)
            debate_id = "dbt-" + secrets.token_hex(4)

        # Broadcast the debate to all experts
        app.verification_mgr.broadcast(
            session_id,
            "system",
            f"Debate started: {topic} (debate_id={debate_id}). "
            f"Use kb_propose, kb_critique, kb_vote to participate.",
        )

        return json.dumps({
            "debate_id": debate_id,
            "topic": topic,
            "context": context,
            "source_tool": "spec",
            "instructions": (
                f"Experts should use kb_propose(debate_id='{debate_id}', ...) to propose "
                f"interpretations with datasheet page references. "
                f"Use kb_critique() to challenge proposals. "
                f"Use kb_vote() to vote. "
                f"Use kb_resolve_debate('{debate_id}') when voting is complete."
            ),
        }, indent=2)

    # ── Verification Summary ──────────────────────────────────────────

    @mcp.tool(
        name="spec_verification_summary",
        description=(
            "Get full verification summary: confirmed, disputed, corrected, "
            "unverified items."
        ),
    )
    def _spec_verification_summary(
        session_id: str,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)
        summary = app.verification_mgr.get_summary(session_id)
        return json.dumps(summary, indent=2)

    # ── Orchestrator ──────────────────────────────────────────────────

    @mcp.tool(
        name="orchestrate_verification",
        description=(
            "Plan a multi-agent spec verification. Returns step-by-step instructions "
            "for AI agents to cross-check all extracted specs. Agents read the original "
            "documents and verify every register, pin, protocol, and timing value."
        ),
    )
    def _orchestrate_verification(
        session_id: str,
        max_agents: int = 5,
        ctx: Optional[Context] = None,
    ) -> str:
        app = _get_app(ctx)

        # 1. Get specs from the spec session
        spec_session = app.store.get_session(session_id)
        specs = spec_session.specs
        if not specs:
            return json.dumps({"error": "No specs in session. Ingest or add specs first."})

        # 2. Start a verification session
        vsess_id = app.verification_mgr.start_session(
            project_path=spec_session.project_path,
            name=f"verification-{session_id}",
        )

        # 3. Analyze specs to determine agent roles
        components_info: list[str] = []
        total_registers = 0
        total_pins = 0
        total_protocols = 0
        total_timing = 0
        protocols_used: set[str] = set()
        categories_used: set[str] = set()

        for s in specs:
            reg_count = len(s.registers)
            pin_count = len(s.pins)
            proto_count = len(s.protocols)
            timing_count = len(s.timing)
            total_registers += reg_count
            total_pins += pin_count
            total_protocols += proto_count
            total_timing += timing_count

            parts = [f"{s.name or s.id}"]
            if reg_count:
                parts.append(f"{reg_count} registers")
            if pin_count:
                parts.append(f"{pin_count} pins")
            if proto_count:
                parts.append(f"{proto_count} protocols")
            if timing_count:
                parts.append(f"{timing_count} timing constraints")
            components_info.append(f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0])

            if s.category:
                categories_used.add(s.category.lower())
            for p in s.protocols:
                if p.protocol:
                    protocols_used.add(p.protocol.upper())

        # 4. Determine expert agents based on spec content
        agents: list[dict] = []
        agent_count = 0

        # Always include MCU/register expert if registers present
        if total_registers > 0 and agent_count < max_agents:
            agents.append({
                "role": "mcu-peripherals",
                "focus": "Register addresses, field names, reset values, access modes, clock tree",
            })
            agent_count += 1

        # Communication protocols expert
        if total_protocols > 0 and agent_count < max_agents:
            agents.append({
                "role": "communication-protocols",
                "focus": f"Protocol configuration: {', '.join(sorted(protocols_used))} -- speeds, modes, pin assignments",
            })
            agent_count += 1

        # Timing expert
        if total_timing > 0 and agent_count < max_agents:
            agents.append({
                "role": "timing-constraints",
                "focus": "All timing values match datasheet -- setup/hold times, clock frequencies, delays",
            })
            agent_count += 1

        # Pin configuration expert
        if total_pins > 0 and agent_count < max_agents:
            agents.append({
                "role": "pin-configuration",
                "focus": "Pin assignments, alternate functions, pull-up/down, drive strength",
            })
            agent_count += 1

        # Power/memory expert
        has_power = any(len(s.power) > 0 for s in specs)
        has_memory = any(len(s.memory_map) > 0 for s in specs)
        if (has_power or has_memory) and agent_count < max_agents:
            agents.append({
                "role": "power-memory",
                "focus": "Power supply rails, voltage ranges, current limits, memory map regions and sizes",
            })
            agent_count += 1

        # 5. Build source documents list
        source_docs = sorted({s.source_doc for s in specs if s.source_doc})

        # 6. Build agent instructions for each phase
        # Phase 1: Independent Verification
        phase1_instructions: list[dict] = []
        for agent in agents:
            role = agent["role"]
            docs_note = ""
            if source_docs:
                docs_note = (
                    f" Use kb_read_document to read the original PDFs: {', '.join(source_docs)}."
                )

            phase1_instructions.append({
                "agent_role": role,
                "description": (
                    f"Read the ORIGINAL DOCUMENT (not just extracted data).{docs_note} "
                    f"Focus on: {agent['focus']}. "
                    f"For EVERY field in your domain, call spec_claim_component() first, "
                    f"then call spec_verify() for each field with status='confirm', 'dispute', "
                    f"or 'correct'. For corrections, include original_value (the extracted "
                    f"value) and corrected_value (the correct value from the datasheet). "
                    f"Reference specific page numbers, table numbers, and "
                    f"figure numbers as evidence. "
                    f"Call spec_mark_phase_done(phase=1) when complete."
                ),
                "tools_to_use": [
                    "spec_claim_component", "spec_verify",
                    "kb_read_document", "spec_mark_phase_done",
                ],
            })

        # Phase 2: Cross-Check
        phase2_instructions: list[dict] = []
        for agent in agents:
            role = agent["role"]
            phase2_instructions.append({
                "agent_role": role,
                "description": (
                    f"Read ALL verifications from Phase 1 via spec_get_verifications(). "
                    f"Cross-check other experts' work: do timing values in protocols match "
                    f"timing constraints? Do pin speeds support the configured clock rates? "
                    f"Are register addresses consistent with the memory map? "
                    f"Flag any inconsistency via spec_verify(status='dispute') with evidence. "
                    f"Use spec_send_message() to ask other experts clarifying questions. "
                    f"Call spec_mark_phase_done(phase=2) when complete."
                ),
                "tools_to_use": [
                    "spec_get_verifications", "spec_verify",
                    "spec_send_message", "spec_mark_phase_done",
                ],
            })

        # Phase 3: Resolve Disputes
        phase3_instructions = [{
            "description": (
                "Check spec_verification_status() for disputes. For each dispute, "
                "start spec_start_debate() with the topic and context from the dispute. "
                "Experts propose interpretations with page references via kb_propose(). "
                "Critique each other's proposals via kb_critique(). "
                "Vote via kb_vote(). "
                "Resolve via kb_resolve_debate() when voting is complete. "
                "Call spec_mark_phase_done(phase=3) when all disputes are resolved."
            ),
            "tools_to_use": [
                "spec_verification_status", "spec_start_debate",
                "kb_propose", "kb_critique", "kb_vote",
                "kb_resolve_debate", "spec_mark_phase_done",
            ],
        }]

        # Phase 4: Generate Report
        phase4_instructions = [{
            "description": (
                "Call spec_verification_summary() to confirm all items are verified. "
                "Then call spec_generate_report() to produce the final verified "
                "specification report. The report will include a Verification Status "
                "section showing confirmed, corrected, and resolved disputes."
            ),
            "tools_to_use": [
                "spec_verification_summary", "spec_generate_report",
            ],
        }]

        plan = {
            "verification_session": vsess_id,
            "spec_session": session_id,
            "components_to_verify": components_info,
            "agents": agents,
            "phases": [
                {
                    "phase": 1,
                    "name": "Independent Verification",
                    "description": (
                        "Each expert reads the ORIGINAL DOCUMENT (not just extracted data) "
                        "and verifies fields in their domain."
                    ),
                    "instructions": phase1_instructions,
                },
                {
                    "phase": 2,
                    "name": "Cross-Check",
                    "description": (
                        "Each expert reviews OTHER experts' verifications. Look for: "
                        "missed errors, disagreements, inconsistencies between components."
                    ),
                    "instructions": phase2_instructions,
                },
                {
                    "phase": 3,
                    "name": "Resolve Disputes",
                    "description": (
                        "If any disputes exist, start a debate. All experts participate."
                    ),
                    "instructions": phase3_instructions,
                },
                {
                    "phase": 4,
                    "name": "Generate Report",
                    "description": (
                        "After all verifications, generate the final verified spec report."
                    ),
                    "instructions": phase4_instructions,
                },
            ],
            "summary": (
                f"Verification session ready. {len(agents)} agents will cross-check "
                f"{total_registers} registers, {total_pins} pins, "
                f"{total_protocols} protocols, {total_timing} timing constraints "
                f"across {len(specs)} components. Execute 4 phases in order."
            ),
        }
        return json.dumps(plan, indent=2)

    return mcp
