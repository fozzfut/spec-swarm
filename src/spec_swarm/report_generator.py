"""Generate structured Specification Reports from SpecSwarm sessions."""

import re
from datetime import datetime, timezone


def generate_report(
    specs: list,
    session_id: str,
    project_name: str = "",
    verification_summary: dict = None,
) -> str:
    """Build a markdown spec report from a list of HardwareSpec objects.

    Parameters
    ----------
    specs : list
        List of HardwareSpec dataclass instances.
    session_id : str
        Session identifier.
    project_name : str
        Optional project name for the report title.
    verification_summary : dict, optional
        Output from SpecSessionManager.get_summary(). When provided, a
        Verification Status section is appended after Architectural Constraints.

    Returns
    -------
    str
        Complete markdown report.
    """
    if not project_name:
        project_name = session_id

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Specification Report: {project_name}")
    lines.append(f"Generated: {date_str}")
    lines.append(f"Session: {session_id}")
    lines.append("")

    # Gather aggregates
    total_registers = sum(len(s.registers) for s in specs)
    total_pins = sum(len(s.pins) for s in specs)
    all_protocols: list[str] = []
    for s in specs:
        for p in s.protocols:
            proto_name = p.protocol.upper()
            if proto_name and proto_name not in all_protocols:
                all_protocols.append(proto_name)

    # ── Section 1: System Overview ────────────────────────────────────
    lines.append("## 1. System Overview")
    component_names = [s.name or s.id for s in specs]
    lines.append(f"- Components: {', '.join(component_names) if component_names else 'None'}")
    lines.append(f"- Total registers: {total_registers}")
    lines.append(f"- Total pins configured: {total_pins}")
    lines.append(f"- Protocols: {', '.join(all_protocols) if all_protocols else 'None'}")
    lines.append("")

    # ── Section 2: Components ─────────────────────────────────────────
    lines.append("## 2. Components")
    lines.append("")

    for idx, spec in enumerate(specs, start=1):
        comp_name = spec.name or spec.id
        cat_label = f" ({spec.category})" if spec.category else ""
        lines.append(f"### 2.{idx} {comp_name}{cat_label}")
        lines.append(f"- Category: {spec.category or 'unknown'}")
        if spec.source_doc:
            lines.append(f"- Source: {spec.source_doc}")
        if spec.manufacturer:
            lines.append(f"- Manufacturer: {spec.manufacturer}")
        if spec.part_number:
            lines.append(f"- Part number: {spec.part_number}")
        lines.append("")

        # Registers
        if spec.registers:
            lines.append(f"#### Registers ({len(spec.registers)} total)")
            lines.append("| Name | Address | Size | Access | Description |")
            lines.append("|------|---------|------|--------|-------------|")
            for reg in spec.registers:
                lines.append(
                    f"| {reg.name} | {reg.address} | {reg.size_bits} "
                    f"| {reg.access} | {reg.description} |"
                )
            lines.append("")

        # Pins
        if spec.pins:
            lines.append("#### Pin Configuration")
            lines.append("| Pin | Function | AF | Direction | Notes |")
            lines.append("|-----|----------|----|-----------|-------|")
            for pin in spec.pins:
                af = str(pin.af_number) if pin.af_number >= 0 else ""
                lines.append(
                    f"| {pin.pin} | {pin.function} | {af} "
                    f"| {pin.direction} | {pin.notes} |"
                )
            lines.append("")

    # ── Section 3: Communication Protocols ────────────────────────────
    lines.append("## 3. Communication Protocols")
    lines.append("")

    proto_idx = 0
    for spec in specs:
        for proto in spec.protocols:
            proto_idx += 1
            role_label = f" ({proto.role.title()})" if proto.role else ""
            lines.append(f"### 3.{proto_idx} {proto.instance or proto.protocol}{role_label}")
            if proto.speed:
                lines.append(f"- Clock/Speed: {proto.speed}")
            if proto.mode:
                lines.append(f"- Mode: {proto.mode}")
            if proto.pins:
                pin_strs = [f"{p.pin} ({p.function})" for p in proto.pins]
                lines.append(f"- Pins: {', '.join(pin_strs)}")
            if proto.notes:
                lines.append(f"- Notes: {proto.notes}")
            lines.append(f"- Component: {spec.name or spec.id}")
            lines.append("")

    if proto_idx == 0:
        lines.append("No protocol configurations found.")
        lines.append("")

    # ── Section 4: Timing Constraints ─────────────────────────────────
    lines.append("## 4. Timing Constraints")
    lines.append("")

    all_timing = []
    for spec in specs:
        for t in spec.timing:
            all_timing.append((spec.name or spec.id, t))

    if all_timing:
        lines.append("| Parameter | Min | Typ | Max | Unit | Source | Critical |")
        lines.append("|-----------|-----|-----|-----|------|--------|----------|")
        for comp, t in all_timing:
            crit = "Yes" if t.critical else "No"
            lines.append(
                f"| {t.parameter} | {t.min_value or '-'} | {t.typ_value or '-'} "
                f"| {t.max_value or '-'} | {t.unit} | {t.source or comp} | {crit} |"
            )
        lines.append("")
    else:
        lines.append("No timing constraints found.")
        lines.append("")

    # ── Section 5: Power Budget ───────────────────────────────────────
    lines.append("## 5. Power Budget")
    lines.append("")

    all_power = []
    for spec in specs:
        for p in spec.power:
            all_power.append((spec.name or spec.id, p))

    if all_power:
        lines.append("| Rail | Min V | Typ V | Max V | Max I | Notes |")
        lines.append("|------|-------|-------|-------|-------|-------|")
        for comp, p in all_power:
            notes = p.notes or comp
            lines.append(
                f"| {p.rail} | {p.min_voltage or '-'} | {p.typ_voltage or '-'} "
                f"| {p.max_voltage or '-'} | {p.max_current or '-'} | {notes} |"
            )
        lines.append("")
    else:
        lines.append("No power specifications found.")
        lines.append("")

    # ── Section 6: Memory Map ─────────────────────────────────────────
    lines.append("## 6. Memory Map")
    lines.append("")

    all_memory = []
    for spec in specs:
        for m in spec.memory_map:
            all_memory.append((spec.name or spec.id, m))

    if all_memory:
        lines.append("| Region | Start | End | Size | Access | Notes |")
        lines.append("|--------|-------|-----|------|--------|-------|")
        for comp, m in all_memory:
            desc = m.description or comp
            lines.append(
                f"| {m.name} | {m.start_address or '-'} | {m.end_address or '-'} "
                f"| {m.size or '-'} | {m.access or '-'} | {desc} |"
            )
        lines.append("")
    else:
        lines.append("No memory map regions found.")
        lines.append("")

    # ── Section 7: Conflicts & Warnings ───────────────────────────────
    warnings = _detect_warnings(specs)
    lines.append("## 7. Conflicts & Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("No conflicts or warnings detected.")
        lines.append("")

    # ── Section 8: Architectural Constraints ──────────────────────────
    arch_constraints = extract_arch_constraints(specs)
    lines.append("## 8. Architectural Constraints")
    lines.append("These constraints MUST be respected by the software architecture:")
    lines.append("")
    if arch_constraints:
        for i, c in enumerate(arch_constraints, start=1):
            lines.append(f"{i}. {c}")
        lines.append("")
    else:
        lines.append("No architectural constraints derived.")
        lines.append("")

    # ── Section 9: Verification Status ────────────────────────────────
    if verification_summary is not None:
        total_v = verification_summary.get("total_verifications", 0)
        confirmed_v = verification_summary.get("confirmed", 0)
        disputed_v = verification_summary.get("disputed", 0)
        corrected_v = verification_summary.get("corrected", 0)
        conf_rate = verification_summary.get("confirmation_rate", 0.0)

        lines.append("## 9. Verification Status")
        lines.append(f"- Fields checked: {total_v}")
        if total_v > 0:
            pct = conf_rate
            lines.append(f"  - Confirmed: {confirmed_v} ({pct}%)")
            lines.append(f"  - Corrected: {corrected_v}")
            lines.append(f"  - Disputed: {disputed_v}")
        else:
            lines.append("- No verification data available.")
        lines.append("")

        # Corrections Applied table
        corrections = verification_summary.get("corrections", [])
        if corrections:
            lines.append("### Corrections Applied")
            lines.append("| Field | Original | Corrected | Expert | Evidence |")
            lines.append("|-------|----------|-----------|--------|----------|")
            for c in corrections:
                field_name = c.get("field", "")
                old_val = c.get("old_value", "")
                new_val = c.get("new_value", "")
                expert = c.get("expert", "")
                evidence = c.get("evidence", "")
                lines.append(f"| {field_name} | {old_val} | {new_val} | {expert} | {evidence} |")
            lines.append("")

        # Disputes table
        disputes = verification_summary.get("disputes", [])
        if disputes:
            lines.append("### Disputes")
            lines.append("| Field | Expert | Evidence |")
            lines.append("|-------|--------|----------|")
            for d in disputes:
                field_name = d.get("field", "")
                expert = d.get("expert", "")
                evidence = d.get("evidence", "")
                lines.append(f"| {field_name} | {expert} | {evidence} |")
            lines.append("")

        # Experts involved
        experts = verification_summary.get("experts_involved", [])
        if experts:
            lines.append(f"### Verification Experts: {', '.join(experts)}")
            lines.append("")

    return "\n".join(lines)


def extract_arch_constraints(specs: list) -> list[str]:
    """Analyze specs and derive architectural constraints.

    Rules:
    - Critical timing constraints -> "Real-time scheduling required for X"
    - Shared SPI/I2C bus -> "Mutex required for bus X shared between A and B"
    - Memory limits -> "Total firmware must fit in N flash"
    - Power modes -> "Sleep mode incompatible with peripheral Y always-on"
    - Pin conflicts -> "Resolve pin X conflict before implementation"
    """
    constraints: list[str] = []

    # 1. Critical timing -> scheduling constraints
    for spec in specs:
        comp = spec.name or spec.id
        for t in spec.timing:
            if t.critical:
                if t.max_value:
                    constraints.append(
                        f"{t.parameter} < {t.max_value} {t.unit} "
                        f"({comp}) -> real-time task scheduling required"
                    )
                elif t.min_value:
                    constraints.append(
                        f"{t.parameter} >= {t.min_value} {t.unit} "
                        f"({comp}) -> minimum delay must be enforced"
                    )

    # 2. Shared buses -> concurrency constraints
    bus_users: dict[str, list[str]] = {}  # bus_instance -> [component names]
    for spec in specs:
        comp = spec.name or spec.id
        for proto in spec.protocols:
            instance = proto.instance.upper()
            if instance:
                if instance not in bus_users:
                    bus_users[instance] = []
                if comp not in bus_users[instance]:
                    bus_users[instance].append(comp)

    for bus, users in bus_users.items():
        if len(users) > 1:
            constraints.append(
                f"{bus} shared between {' and '.join(users)} "
                f"-> mutex or task separation required"
            )

    # 3. Memory limits -> size constraints
    for spec in specs:
        comp = spec.name or spec.id
        for region in spec.memory_map:
            if region.size and region.name:
                name_lower = region.name.lower()
                if "flash" in name_lower:
                    constraints.append(
                        f"{region.name} {region.size} ({comp}) "
                        f"-> firmware + bootloader + OTA partition must fit"
                    )
                elif "sram" in name_lower or "ram" in name_lower:
                    constraints.append(
                        f"{region.name} {region.size} ({comp}) "
                        f"-> stack + heap + DMA buffers must fit"
                    )

    # 4. Power constraints -> power mode constraints
    for spec in specs:
        comp = spec.name or spec.id
        for pwr in spec.power:
            if pwr.max_current:
                # Parse current value
                match = re.search(r"(\d+(?:\.\d+)?)\s*(mA|uA|A)", pwr.max_current)
                if match:
                    constraints.append(
                        f"Power rail {pwr.rail} max {pwr.max_current} ({comp}) "
                        f"-> sleep mode or power gating may be needed"
                    )

    # 5. Pin conflicts -> resolution requirements
    pin_usage: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        comp = spec.name or spec.id
        for pin in spec.pins:
            key = pin.pin.upper()
            if not key:
                continue
            if key not in pin_usage:
                pin_usage[key] = []
            pin_usage[key].append((comp, pin.function))

    for pin_name, usages in pin_usage.items():
        if len(usages) > 1:
            functions = set(func for _, func in usages)
            if len(functions) > 1:
                usage_strs = [f"{comp}/{func}" for comp, func in usages]
                constraints.append(
                    f"Pin {pin_name} conflict: {', '.join(usage_strs)} "
                    f"-> resolve before implementation"
                )

    # 6. Protocol speed constraints -> throughput limits
    for spec in specs:
        comp = spec.name or spec.id
        for proto in spec.protocols:
            if proto.protocol.upper() == "I2C" and proto.speed:
                # I2C has inherent transaction rate limits
                speed_match = re.search(r"(\d+)\s*(kHz|MHz)", proto.speed)
                if speed_match:
                    val = int(speed_match.group(1))
                    unit = speed_match.group(2)
                    if unit == "kHz" and val <= 400:
                        constraints.append(
                            f"{proto.instance} at {proto.speed} "
                            f"-> max ~{val // 16} transactions/second"
                        )

    # 7. Free-form datasheet constraints
    for spec in specs:
        for c_text in spec.constraints:
            constraints.append(c_text)

    return constraints


def _detect_warnings(specs: list) -> list[str]:
    """Detect conflicts and warnings from spec data."""
    warnings: list[str] = []

    # Pin collisions
    pin_usage: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        comp = spec.name or spec.id
        for pin in spec.pins:
            key = pin.pin.upper()
            if not key:
                continue
            if key not in pin_usage:
                pin_usage[key] = []
            pin_usage[key].append((comp, pin.function))

    for pin_name, usages in pin_usage.items():
        functions = set(func for _, func in usages)
        if len(functions) > 1:
            usage_strs = [f"{comp}/{func}" for comp, func in usages]
            warnings.append(
                f"Pin {pin_name} assigned to {', '.join(usage_strs)} -- conflicts with multiple use"
            )

    # Power budget check
    total_consumer_ma = 0.0
    supply_capacity_ma = 0.0
    for spec in specs:
        for pwr in spec.power:
            if not pwr.max_current:
                continue
            match = re.search(r"(\d+(?:\.\d+)?)\s*(mA|uA|A)", pwr.max_current)
            if not match:
                continue
            val = float(match.group(1))
            unit = match.group(2)
            if unit == "uA":
                val /= 1000.0
            elif unit == "A":
                val *= 1000.0
            if spec.category in ("mcu", "power"):
                supply_capacity_ma = max(supply_capacity_ma, val)
            else:
                total_consumer_ma += val

    if supply_capacity_ma > 0 and total_consumer_ma > supply_capacity_ma:
        warnings.append(
            f"Power budget: total estimated draw {total_consumer_ma:.0f}mA, "
            f"max supply {supply_capacity_ma:.0f}mA"
        )

    # I2C address collisions
    i2c_addrs: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        comp = spec.name or spec.id
        for proto in spec.protocols:
            if proto.protocol.upper() == "I2C" and proto.notes:
                addr_match = re.search(r"0x[0-9A-Fa-f]{2}", proto.notes)
                if addr_match:
                    addr = addr_match.group(0).upper()
                    bus = proto.instance
                    key = f"{bus}:{addr}"
                    if key not in i2c_addrs:
                        i2c_addrs[key] = []
                    i2c_addrs[key].append((comp, bus))

    for key, devices in i2c_addrs.items():
        if len(devices) > 1:
            bus_addr = key
            device_names = [d[0] for d in devices]
            warnings.append(
                f"I2C address collision at {bus_addr}: {', '.join(device_names)}"
            )

    return warnings
