"""MCP Server for SpecSwarm -- hardware specification analysis tools."""

import json
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    HardwareSpec, SpecSession, SpecType, Register, PinConfig,
    ProtocolConfig, TimingConstraint, PowerSpec, MemoryRegion, now_iso,
)
from .spec_store import SpecStore
from .expert_profiler import ExpertProfiler


@dataclass
class AppContext:
    store: SpecStore
    profiler: ExpertProfiler


def create_app_context() -> AppContext:
    store = SpecStore()
    profiler = ExpertProfiler()
    return AppContext(store=store, profiler=profiler)


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
                    import re
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
                    import re
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

    return mcp
