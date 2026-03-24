"""Extract structured hardware data from parsed document text.

Uses regex patterns to find:
- Register definitions (address, name, fields, reset values)
- Pin configurations (pin name, alternate functions)
- Protocol parameters (baud rate, clock speed, modes)
- Timing constraints (min/typ/max values with units)
- Memory regions (start address, size)
- Power specifications (voltage ranges, current limits)
"""

from __future__ import annotations

import re


def extract_registers(text: str) -> list[dict]:
    """Extract register definitions from text.

    Looks for patterns like:
    - 0x40021000  RCC_CR  Reset: 0x00000083  (Clock control register)
    - Register: GPIO_MODER  Address: 0x48000000  Access: RW
    - ADDR  NAME  RESET  ACCESS  DESCRIPTION (table headers)
    """
    registers: list[dict] = []
    seen_addrs: set[str] = set()

    # Pattern 1: Address followed by register name
    # e.g., "0x40021000  RCC_CR" or "0x4002_1000 RCC_CR"
    pat1 = re.compile(
        r"(0x[0-9A-Fa-f_]{4,12})\s+"            # hex address
        r"([A-Z][A-Z0-9_]{2,40})\b"               # register name (UPPER_CASE)
        r"(?:\s+(?:Reset|RST|Default)[:\s=]*(0x[0-9A-Fa-f]+))?"  # optional reset value
        r"(?:\s+([RrWw/Oo]{1,6}))?"               # optional access type
        r"(?:\s+(.+))?"                            # optional description
    )
    for m in pat1.finditer(text):
        addr = m.group(1).replace("_", "")
        if addr in seen_addrs:
            continue
        seen_addrs.add(addr)
        name = m.group(2)
        reset = m.group(3) or "0x00000000"
        access = (m.group(4) or "rw").lower().replace("/", "")
        desc = (m.group(5) or "").strip()
        registers.append({
            "name": name,
            "address": addr,
            "size_bits": 32,
            "reset_value": reset,
            "access": access,
            "description": desc,
            "fields": [],
        })

    # Pattern 2: "Register: NAME" format (common in reference manuals)
    pat2 = re.compile(
        r"Register[:\s]+([A-Z][A-Z0-9_]{2,40})\s+"
        r"(?:Address|Addr|Offset)[:\s]*(0x[0-9A-Fa-f_]{2,12})"
        r"(?:\s+(?:Access|Type)[:\s]*([A-Za-z/]+))?"
        r"(?:\s+(?:Size|Width)[:\s]*(\d+)\s*(?:bit|b))?"
    , re.IGNORECASE)
    for m in pat2.finditer(text):
        name = m.group(1)
        addr = m.group(2).replace("_", "")
        if addr in seen_addrs:
            continue
        seen_addrs.add(addr)
        access = (m.group(3) or "rw").lower().replace("/", "")
        size = int(m.group(4)) if m.group(4) else 32
        registers.append({
            "name": name,
            "address": addr,
            "size_bits": size,
            "reset_value": "0x00000000",
            "access": access,
            "description": "",
            "fields": [],
        })

    # Build register position index for field assignment
    # Each entry is (text_position, register_index)
    register_positions: list[tuple[int, int]] = []
    reg_idx = 0
    for m in pat1.finditer(text):
        addr = m.group(1).replace("_", "")
        # Find the register with this address
        for i, r in enumerate(registers):
            if r["address"] == addr and i >= reg_idx:
                register_positions.append((m.start(), i))
                reg_idx = i + 1
                break
    for m in pat2.finditer(text):
        addr = m.group(2).replace("_", "")
        for i, r in enumerate(registers):
            if r["address"] == addr and i >= reg_idx:
                register_positions.append((m.start(), i))
                reg_idx = i + 1
                break
    register_positions.sort(key=lambda x: x[0])

    # Extract bit fields for registers already found
    # Pattern: "Bits X:Y  FIELD_NAME  rw  Description" or "Bit X  FIELD_NAME"
    field_pat = re.compile(
        r"Bits?\s+(\d+)(?::(\d+))?\s+"
        r"([A-Z][A-Z0-9_]{1,30})\b"
        r"(?:\s+([rwRW/o]{1,4}))?"
        r"(?:\s+(.+))?"
    )
    for m in field_pat.finditer(text):
        high = int(m.group(1))
        low = int(m.group(2)) if m.group(2) else high
        field_name = m.group(3)
        access = (m.group(4) or "rw").lower().replace("/", "")
        desc = (m.group(5) or "").strip()
        bits = f"{high}:{low}" if high != low else str(high)

        field_info = {
            "name": field_name,
            "bits": bits,
            "access": access,
            "description": desc,
        }

        # Attach to the nearest preceding register based on text position
        field_pos = m.start()
        owner_idx = -1
        for pos, idx in reversed(register_positions):
            if pos <= field_pos:
                owner_idx = idx
                break
        if owner_idx >= 0 and owner_idx < len(registers):
            registers[owner_idx]["fields"].append(field_info)
        elif registers:
            # Fallback: attach to last register if no position match
            registers[-1]["fields"].append(field_info)

    return registers


def extract_pins(text: str) -> list[dict]:
    """Extract pin configurations from text.

    Looks for patterns like:
    - PA0 / UART1_TX / AF7
    - Pin: PB5  Function: SPI1_MOSI  AF: 5
    - GPIO_A, Pin 0 -- ADC1_IN0
    """
    pins: list[dict] = []
    seen_pins: set[str] = set()

    # Pattern 1: Standard MCU pin notation (PA0, PB5, PC13, PD2, etc.)
    # with alternate function
    pat1 = re.compile(
        r"\b(P[A-K]\d{1,2})\b"                      # pin name (PA0-PK15)
        r"(?:\s*[/|,]\s*|\s+)"                        # separator
        r"([A-Z][A-Z0-9_]{2,30})"                     # function name
        r"(?:\s*[/|,]\s*AF\s*(\d{1,2}))?"             # optional AF number
    )
    for m in pat1.finditer(text):
        pin = m.group(1)
        func = m.group(2)
        af = int(m.group(3)) if m.group(3) else -1

        key = f"{pin}:{func}"
        if key in seen_pins:
            continue
        seen_pins.add(key)

        # Infer direction from function name
        direction = _infer_pin_direction(func)

        pins.append({
            "pin": pin,
            "function": func,
            "af_number": af,
            "direction": direction,
            "pull": "",
            "speed": "",
            "notes": "",
        })

    # Pattern 2: "Pin: XX  Function: YY" format
    pat2 = re.compile(
        r"Pin[:\s]+([A-Z0-9_]{2,10})\s+"
        r"(?:Function|Func|Signal)[:\s]+([A-Z][A-Z0-9_]{2,30})"
        r"(?:\s+AF[:\s]*(\d{1,2}))?"
    , re.IGNORECASE)
    for m in pat2.finditer(text):
        pin = m.group(1)
        func = m.group(2)
        af = int(m.group(3)) if m.group(3) else -1

        key = f"{pin}:{func}"
        if key in seen_pins:
            continue
        seen_pins.add(key)

        direction = _infer_pin_direction(func)
        pins.append({
            "pin": pin,
            "function": func,
            "af_number": af,
            "direction": direction,
            "pull": "",
            "speed": "",
            "notes": "",
        })

    # Pattern 3: ESP32-style pin notation (GPIO0, GPIO2, IO0, etc.)
    # Exclude GPIO state keywords that describe pin levels, not functions
    _GPIO_STATE_KEYWORDS = {
        "HIGH", "LOW", "ON", "OFF", "SET", "CLEAR", "RESET",
        "PULSE", "TOGGLE", "TRUE", "FALSE", "ACTIVE", "INACTIVE",
    }
    pat3 = re.compile(
        r"\b((?:GPIO|IO)\d{1,2})\b"
        r"(?:\s*[/|,]\s*|\s+)"
        r"([A-Z][A-Z0-9_]{2,30})"
    )
    for m in pat3.finditer(text):
        pin = m.group(1)
        func = m.group(2)

        # Skip state keywords -- they describe pin levels, not functions
        if func in _GPIO_STATE_KEYWORDS:
            continue
        # Skip references to other GPIO pins (e.g. "GPIO5 GPIO9" in sequences)
        if re.match(r"^(?:GPIO|IO)\d{1,2}$", func):
            continue

        key = f"{pin}:{func}"
        if key in seen_pins:
            continue
        seen_pins.add(key)

        direction = _infer_pin_direction(func)
        pins.append({
            "pin": pin,
            "function": func,
            "af_number": -1,
            "direction": direction,
            "pull": "",
            "speed": "",
            "notes": "",
        })

    return pins


def _infer_pin_direction(function: str) -> str:
    """Infer pin direction from function name."""
    func_upper = function.upper()
    if any(kw in func_upper for kw in ("_TX", "_MOSI", "_SCK", "_OUT", "PWM", "LED")):
        return "output"
    if any(kw in func_upper for kw in ("_RX", "_MISO", "_IN", "BTN", "BUTTON", "EXT")):
        return "input"
    if any(kw in func_upper for kw in ("ADC", "DAC", "AIN", "AOUT")):
        return "analog"
    if any(kw in func_upper for kw in ("_SDA", "_SCL", "_CK", "_NSS", "_CS")):
        return "alternate"
    return ""


def extract_protocols(text: str) -> list[dict]:
    """Extract communication protocol configurations from text.

    Looks for patterns like:
    - SPI1: Mode 0, 10 MHz, MSB first
    - I2C address: 0x68, 400 kHz
    - UART: 115200 baud, 8N1
    - CAN: 500 kbps, 11-bit ID
    """
    protocols: list[dict] = []
    seen: set[str] = set()

    # SPI configuration
    spi_pat = re.compile(
        r"\b(SPI\d?)\b[:\s]+"
        r"(?:.*?(?:Mode|CPOL/CPHA)[:\s]*(\d)[,;\s]*)?"
        r"(?:.*?(\d+(?:\.\d+)?)\s*(?:MHz|Mbps|kHz)[,;\s]*)?"
        r"(?:.*?(MSB|LSB)\s*first)?"
    , re.IGNORECASE)
    for m in spi_pat.finditer(text):
        instance = m.group(1).upper()
        if instance in seen:
            continue
        seen.add(instance)
        mode = f"Mode {m.group(2)}" if m.group(2) else ""
        speed_val = m.group(3) or ""
        word_order = f"{m.group(4).upper()} first" if m.group(4) else ""

        # Determine speed string from context
        speed = ""
        if speed_val:
            # Find the unit near the speed value
            speed_ctx = text[max(0, m.start(3) - 5):m.end(3) + 10] if m.group(3) else ""
            if "MHz" in speed_ctx or "Mbps" in speed_ctx:
                speed = f"{speed_val} MHz"
            elif "kHz" in speed_ctx or "kbps" in speed_ctx:
                speed = f"{speed_val} kHz"
            else:
                speed = speed_val

        protocols.append({
            "protocol": "SPI",
            "instance": instance,
            "role": "master",
            "speed": speed,
            "mode": mode,
            "data_bits": 8,
            "word_order": word_order,
            "pins": [],
            "notes": "",
        })

    # I2C configuration
    i2c_pat = re.compile(
        r"\b(I2C\d?)\b[:\s]+"
        r"(?:.*?(?:address|addr)[:\s]*(0x[0-9A-Fa-f]{2,4})[,;\s]*)?"
        r"(?:.*?(\d+)\s*(?:kHz|KHz)[,;\s]*)?"
    , re.IGNORECASE)
    for m in i2c_pat.finditer(text):
        instance = m.group(1).upper()
        if instance in seen:
            continue
        seen.add(instance)
        addr = m.group(2) or ""
        speed = f"{m.group(3)} kHz" if m.group(3) else ""
        notes = f"Device address: {addr}" if addr else ""

        protocols.append({
            "protocol": "I2C",
            "instance": instance,
            "role": "master",
            "speed": speed,
            "mode": "",
            "data_bits": 8,
            "word_order": "",
            "pins": [],
            "notes": notes,
        })

    # UART configuration
    uart_pat = re.compile(
        r"\b(U(?:S)?ART\d?|UART\d?)\b[:\s]+"
        r"(?:.*?(\d{4,7})\s*(?:baud|bps|Baud)[,;\s]*)?"
        r"(?:.*?(\d)([NnEeOo])(\d)[,;\s]*)?"
    , re.IGNORECASE)
    for m in uart_pat.finditer(text):
        instance = m.group(1).upper()
        if instance in seen:
            continue
        seen.add(instance)
        baud = m.group(2) or ""
        speed = f"{baud} baud" if baud else ""
        data_bits = int(m.group(3)) if m.group(3) else 8
        parity = m.group(4).upper() if m.group(4) else "N"
        stop = m.group(5) if m.group(5) else "1"
        mode = f"{data_bits}{parity}{stop}" if m.group(3) else ""

        protocols.append({
            "protocol": "UART",
            "instance": instance,
            "role": "",
            "speed": speed,
            "mode": mode,
            "data_bits": data_bits,
            "word_order": "",
            "pins": [],
            "notes": "",
        })

    # CAN configuration
    can_pat = re.compile(
        r"\b(CAN\d?|FDCAN\d?)\b[:\s]+"
        r"(?:.*?(\d+)\s*(?:kbps|Kbps|Mbps)[,;\s]*)?"
        r"(?:.*?(11|29|extended|standard)\s*-?\s*bit)?"
    , re.IGNORECASE)
    for m in can_pat.finditer(text):
        instance = m.group(1).upper()
        if instance in seen:
            continue
        seen.add(instance)
        speed_val = m.group(2) or ""
        speed = ""
        if speed_val:
            speed_ctx = text[max(0, m.start(2) - 5):m.end(2) + 10] if m.group(2) else ""
            if "Mbps" in speed_ctx:
                speed = f"{speed_val} Mbps"
            else:
                speed = f"{speed_val} kbps"
        id_type = m.group(3) or ""
        mode = f"{id_type}-bit ID" if id_type and id_type.isdigit() else id_type

        protocols.append({
            "protocol": "CAN",
            "instance": instance,
            "role": "",
            "speed": speed,
            "mode": mode,
            "data_bits": 8,
            "word_order": "",
            "pins": [],
            "notes": "",
        })

    # CAN message IDs and DLC patterns
    # e.g., "CAN ID: 0x181" or "MSG_ID = 0x601" or "CAN_ID(0x200)" or "DLC: 8"
    can_msg_pat = re.compile(
        r"(?:CAN\s*(?:ID|MSG|message)|MSG_ID|COB[_-]?ID|CAN_ID)"
        r"[:\s=(]*"
        r"(0x[0-9A-Fa-f]{1,8})"
        r"(?:\s*[,;)\s]+.*?DLC[:\s=]*(\d))?",
        re.IGNORECASE,
    )
    for m in can_msg_pat.finditer(text):
        msg_id = m.group(1)
        dlc = m.group(2) or "8"
        key = f"CAN_MSG_{msg_id}"
        if key in seen:
            continue
        seen.add(key)

        # Determine standard (11-bit) vs extended (29-bit) from ID value
        id_val = int(msg_id, 16)
        id_mode = "29-bit ID" if id_val > 0x7FF else "11-bit ID"

        protocols.append({
            "protocol": "CAN",
            "instance": f"MSG_{msg_id}",
            "role": "",
            "speed": "",
            "mode": id_mode,
            "data_bits": int(dlc),
            "word_order": "",
            "pins": [],
            "notes": f"CAN message ID {msg_id}, DLC={dlc}",
        })

    # Modbus register addresses and function codes
    # e.g., "holding register 40001" or "register address 0x0064" or "FC03"
    modbus_reg_pat = re.compile(
        r"(?:modbus\s+)?(?:holding|input|coil|discrete)?\s*"
        r"(?:register|reg)\s*"
        r"(?:address)?[:\s=]*((?:0x[0-9A-Fa-f]{1,6})|(?:[0-9]{3,5}))"
        r"(?:\s*[,;]\s*(?:FC|function\s*code)[:\s=]*(\d{1,2}))?"
    , re.IGNORECASE)
    for m in modbus_reg_pat.finditer(text):
        addr = m.group(1)
        fc = m.group(2) or ""
        key = f"MODBUS_REG_{addr}"
        if key in seen:
            continue
        seen.add(key)

        # Infer register type from address range or function code
        reg_type = ""
        if fc:
            fc_int = int(fc)
            if fc_int in (1, 5, 15):
                reg_type = "coil"
            elif fc_int == 2:
                reg_type = "discrete_input"
            elif fc_int in (3, 6, 16):
                reg_type = "holding_register"
            elif fc_int == 4:
                reg_type = "input_register"

        notes = f"Modbus register {addr}"
        if fc:
            notes += f", FC{fc}"
        if reg_type:
            notes += f" ({reg_type})"

        protocols.append({
            "protocol": "Modbus",
            "instance": f"REG_{addr}",
            "role": "",
            "speed": "",
            "mode": f"FC{fc}" if fc else "",
            "data_bits": 16,
            "word_order": "big-endian",
            "pins": [],
            "notes": notes,
        })

    # CANopen Object Dictionary entries (index:subindex)
    # e.g., "OD 0x1018:01" or "index 0x6040 subindex 0x00" or "Object 1018h:01h"
    canopen_od_pat = re.compile(
        r"(?:OD|object\s*dictionary|index|object)\s*"
        r"(?:entry\s*)?"
        r"(0x[0-9A-Fa-f]{4}|[0-9A-Fa-f]{4}[hH])"
        r"(?:\s*[:]\s*|\s+(?:sub(?:index)?)\s*[:=]?\s*)"
        r"(0x[0-9A-Fa-f]{1,2}|[0-9A-Fa-f]{1,2}[hH]|\d{1,3})"
    , re.IGNORECASE)
    for m in canopen_od_pat.finditer(text):
        raw_index = m.group(1)
        raw_sub = m.group(2)

        # Normalize index
        if raw_index.endswith(("h", "H")):
            index_hex = f"0x{raw_index[:-1]}"
        else:
            index_hex = raw_index

        # Normalize subindex
        if raw_sub.endswith(("h", "H")):
            sub_hex = f"0x{raw_sub[:-1]}"
        elif raw_sub.startswith("0x"):
            sub_hex = raw_sub
        else:
            sub_hex = f"0x{int(raw_sub):02X}"

        key = f"CANOPEN_OD_{index_hex}:{sub_hex}"
        if key in seen:
            continue
        seen.add(key)

        # Classify the OD region
        idx_val = int(index_hex, 16)
        if 0x1000 <= idx_val <= 0x1FFF:
            region = "communication"
        elif 0x2000 <= idx_val <= 0x5FFF:
            region = "manufacturer-specific"
        elif 0x6000 <= idx_val <= 0x9FFF:
            region = "standardized-device-profile"
        else:
            region = "other"

        protocols.append({
            "protocol": "CANopen",
            "instance": f"OD_{index_hex}:{sub_hex}",
            "role": "",
            "speed": "",
            "mode": region,
            "data_bits": 0,
            "word_order": "",
            "pins": [],
            "notes": f"CANopen OD entry {index_hex}:{sub_hex} ({region})",
        })

    return protocols


def extract_timing(text: str) -> list[dict]:
    """Extract timing constraints from text.

    Looks for patterns like:
    - t_startup: min 10 us, typ 50 us, max 100 us
    - Conversion time: 12.5 ADC clock cycles (typ)
    - Maximum SPI clock: 10 MHz
    - Watchdog timeout: 32 ms (max)
    """
    timing: list[dict] = []
    seen: set[str] = set()

    # Pattern 1: "parameter: min X, typ Y, max Z" style
    pat1 = re.compile(
        r"(t_\w+|T_\w+|[A-Z][A-Za-z_]+(?:\s+(?:time|delay|period|timeout|latency|jitter)))"
        r"[:\s]+"
        r"(?:(?:min(?:imum)?)[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)[,;\s]*)?"
        r"(?:(?:typ(?:ical)?)[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)[,;\s]*)?"
        r"(?:(?:max(?:imum)?)[:\s]*(\d+(?:\.\d+)?)\s*([a-zA-Z]+))?"
    , re.IGNORECASE)
    for m in pat1.finditer(text):
        param = m.group(1).strip()
        if param in seen:
            continue
        seen.add(param)

        min_val = f"{m.group(2)} {m.group(3)}" if m.group(2) else ""
        typ_val = f"{m.group(4)} {m.group(5)}" if m.group(4) else ""
        max_val = f"{m.group(6)} {m.group(7)}" if m.group(6) else ""

        # Determine unit from whichever value we have
        unit = m.group(3) or m.group(5) or m.group(7) or ""

        timing.append({
            "parameter": param,
            "min_value": min_val,
            "typ_value": typ_val,
            "max_value": max_val,
            "unit": unit,
            "condition": "",
            "source": "",
            "critical": _is_critical_timing(param),
        })

    # Pattern 2: "Maximum/Minimum PARAMETER: VALUE UNIT"
    pat2 = re.compile(
        r"(Maximum|Minimum|Max|Min)\s+"
        r"([A-Za-z][A-Za-z0-9_ ]{2,40}?)"
        r"[:\s]+(\d+(?:\.\d+)?)\s*"
        r"(ns|us|ms|s|MHz|kHz|Hz|GHz|bps|kbps|Mbps)\b"
    , re.IGNORECASE)
    for m in pat2.finditer(text):
        bound = m.group(1).lower()
        param = m.group(2).strip()
        value = f"{m.group(3)} {m.group(4)}"
        unit = m.group(4)

        if param in seen:
            continue
        seen.add(param)

        entry = {
            "parameter": param,
            "min_value": "",
            "typ_value": "",
            "max_value": "",
            "unit": unit,
            "condition": "",
            "source": "",
            "critical": _is_critical_timing(param),
        }
        if bound in ("maximum", "max"):
            entry["max_value"] = value
        else:
            entry["min_value"] = value

        timing.append(entry)

    # Pattern 3: Generic "PARAMETER: VALUE UNIT" with timing-related keywords
    timing_keywords = (
        "startup", "boot", "conversion", "settling", "propagation",
        "rise", "fall", "setup", "hold", "recovery", "pulse",
        "period", "frequency", "clock", "baud", "timeout", "watchdog",
        "deadline", "latency", "jitter", "sampling",
    )
    pat3 = re.compile(
        r"([A-Za-z][A-Za-z0-9_ ]{2,40}?)"
        r"[:\s]+(\d+(?:\.\d+)?)\s*"
        r"(ns|us|ms|s|MHz|kHz|Hz|cycles?)\b"
        r"(?:\s*\((\w+)\))?"  # optional (typ), (max), etc.
    , re.IGNORECASE)
    for m in pat3.finditer(text):
        param = m.group(1).strip()
        value_str = f"{m.group(2)} {m.group(3)}"
        unit = m.group(3)
        qualifier = (m.group(4) or "").lower()

        # Only process if param contains a timing keyword
        param_lower = param.lower()
        if not any(kw in param_lower for kw in timing_keywords):
            continue

        if param in seen:
            continue
        seen.add(param)

        entry = {
            "parameter": param,
            "min_value": "",
            "typ_value": "",
            "max_value": "",
            "unit": unit,
            "condition": "",
            "source": "",
            "critical": _is_critical_timing(param),
        }
        if qualifier == "max":
            entry["max_value"] = value_str
        elif qualifier == "min":
            entry["min_value"] = value_str
        else:
            entry["typ_value"] = value_str

        timing.append(entry)

    return timing


def _is_critical_timing(parameter: str) -> bool:
    """Determine if a timing parameter is critical (violation causes damage/malfunction)."""
    critical_keywords = (
        "maximum", "absolute", "overcurrent", "overvoltage",
        "dead_time", "dead time", "deadtime",
        "watchdog", "wdt", "brown", "power_on",
        "esd", "surge", "short_circuit",
    )
    param_lower = parameter.lower()
    return any(kw in param_lower for kw in critical_keywords)


def extract_power(text: str) -> list[dict]:
    """Extract power specifications from text.

    Looks for patterns like:
    - VDD: 2.7V to 3.6V (typ 3.3V)
    - Supply current: 150 mA (max)
    - VDDA: 2.4V min, 3.6V max
    """
    power: list[dict] = []
    seen_rails: set[str] = set()

    # Pattern 1: Voltage rail with range
    # e.g., "VDD: 2.7V to 3.6V" or "VDD = 3.3V (2.7V - 3.6V)"
    pat1 = re.compile(
        r"\b(V(?:DD|CC|BAT|DDA|REF|BUS|IN|OUT|IO|SS|EE)[A-Z0-9]*)\b"
        r"[:\s=]+"
        r"(?:(\d+(?:\.\d+)?)\s*V?\s*(?:to|[-~])\s*(\d+(?:\.\d+)?)\s*V)"
        r"(?:\s*\(?\s*(?:typ(?:ical)?[:\s]*)?(\d+(?:\.\d+)?)\s*V\)?)?"
    , re.IGNORECASE)
    for m in pat1.finditer(text):
        rail = m.group(1).upper()
        if rail in seen_rails:
            continue
        seen_rails.add(rail)

        min_v = f"{m.group(2)}V"
        max_v = f"{m.group(3)}V"
        typ_v = f"{m.group(4)}V" if m.group(4) else ""

        power.append({
            "rail": rail,
            "min_voltage": min_v,
            "typ_voltage": typ_v,
            "max_voltage": max_v,
            "max_current": "",
            "notes": "",
        })

    # Pattern 2: "RAIL = VALUE V" simple assignment
    pat2 = re.compile(
        r"\b(V(?:DD|CC|BAT|DDA|REF|BUS|IN|OUT|IO|SS|EE)[A-Z0-9]*)\b"
        r"\s*[=:]\s*"
        r"(\d+(?:\.\d+)?)\s*V\b"
    , re.IGNORECASE)
    for m in pat2.finditer(text):
        rail = m.group(1).upper()
        if rail in seen_rails:
            continue
        seen_rails.add(rail)
        typ_v = f"{m.group(2)}V"
        power.append({
            "rail": rail,
            "min_voltage": "",
            "typ_voltage": typ_v,
            "max_voltage": "",
            "max_current": "",
            "notes": "",
        })

    # Pattern 3: Current consumption
    current_pat = re.compile(
        r"(?:supply|operating|standby|sleep|quiescent|max(?:imum)?)\s+"
        r"current[:\s]+(\d+(?:\.\d+)?)\s*(uA|mA|A)\b"
    , re.IGNORECASE)
    for m in current_pat.finditer(text):
        current_val = f"{m.group(1)} {m.group(2)}"
        # Attach to the last power rail or create a generic entry
        if power:
            if not power[-1]["max_current"]:
                power[-1]["max_current"] = current_val
            else:
                power[-1]["notes"] += f" Additional current spec: {current_val}"
        else:
            power.append({
                "rail": "VDD",
                "min_voltage": "",
                "typ_voltage": "",
                "max_voltage": "",
                "max_current": current_val,
                "notes": "Auto-extracted current specification",
            })

    return power


def extract_memory_map(text: str) -> list[dict]:
    """Extract memory map regions from text.

    Looks for patterns like:
    - Flash: 0x08000000 - 0x0807FFFF (512 KB)
    - SRAM1: 0x20000000, 128 KB
    - Peripheral base: 0x40000000
    """
    regions: list[dict] = []
    seen_addrs: set[str] = set()

    # Pattern 1: "Name: START - END (SIZE)"
    pat1 = re.compile(
        r"\b(Flash|FLASH|SRAM\d?|RAM|ROM|EEPROM|CCM|ITCM|DTCM|"
        r"Peripheral|APB\d?|AHB\d?|System|Boot|OTP|Option|Backup|"
        r"SDRAM|QSPI|FSMC|FMC|External|Internal)\b"
        r"[:\s]+"
        r"(0x[0-9A-Fa-f]{4,12})\s*"
        r"(?:[-~]|to)\s*"
        r"(0x[0-9A-Fa-f]{4,12})"
        r"(?:\s*\(?\s*(\d+(?:\.\d+)?)\s*(KB|MB|GB|B|kB|bytes)\s*\)?)?"
    , re.IGNORECASE)
    for m in pat1.finditer(text):
        name = m.group(1)
        start = m.group(2)
        end = m.group(3)

        if start in seen_addrs:
            continue
        seen_addrs.add(start)

        size = f"{m.group(4)} {m.group(5).upper()}" if m.group(4) else ""
        access = _infer_memory_access(name)

        regions.append({
            "name": name,
            "start_address": start,
            "end_address": end,
            "size": size,
            "access": access,
            "description": "",
        })

    # Pattern 2: "Name: START, SIZE"
    pat2 = re.compile(
        r"\b(Flash|FLASH|SRAM\d?|RAM|ROM|EEPROM|CCM|ITCM|DTCM|"
        r"Peripheral|System|Boot|OTP|Backup|SDRAM|QSPI)\b"
        r"[:\s]+"
        r"(0x[0-9A-Fa-f]{4,12})"
        r"[,;\s]+"
        r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|B|kB|bytes)\b"
    , re.IGNORECASE)
    for m in pat2.finditer(text):
        name = m.group(1)
        start = m.group(2)

        if start in seen_addrs:
            continue
        seen_addrs.add(start)

        size = f"{m.group(3)} {m.group(4).upper()}"
        access = _infer_memory_access(name)

        regions.append({
            "name": name,
            "start_address": start,
            "end_address": "",
            "size": size,
            "access": access,
            "description": "",
        })

    return regions


def _infer_memory_access(name: str) -> str:
    """Infer memory access type from region name."""
    name_upper = name.upper()
    if name_upper in ("FLASH", "ROM", "BOOT", "OTP"):
        return "rx"
    if name_upper in ("SRAM", "SRAM1", "SRAM2", "RAM", "CCM", "ITCM", "DTCM",
                       "SDRAM", "BACKUP", "EEPROM"):
        return "rw"
    if name_upper in ("PERIPHERAL", "APB1", "APB2", "AHB1", "AHB2"):
        return "rw"
    return "rw"


def extract_all(text: str, component_name: str = "") -> dict:
    """Extract all hardware specification data from text.

    Returns a dict with all extracted data that can be used to create a HardwareSpec.
    """
    registers = extract_registers(text)
    pins = extract_pins(text)
    protocols = extract_protocols(text)
    timing_list = extract_timing(text)
    power_list = extract_power(text)
    memory = extract_memory_map(text)

    # Auto-detect category from content
    category = _detect_category(text, registers, pins, protocols)

    # Extract free-form constraints
    constraints = _extract_constraints(text)

    return {
        "name": component_name,
        "category": category,
        "registers": registers,
        "pins": pins,
        "protocols": protocols,
        "timing": timing_list,
        "power": power_list,
        "memory_map": memory,
        "constraints": constraints,
        "extraction_stats": {
            "registers_found": len(registers),
            "pins_found": len(pins),
            "protocols_found": len(protocols),
            "timing_constraints_found": len(timing_list),
            "power_specs_found": len(power_list),
            "memory_regions_found": len(memory),
            "constraints_found": len(constraints),
        },
    }


def _detect_category(text: str, registers: list, pins: list, protocols: list) -> str:
    """Auto-detect component category from extracted data."""
    text_lower = text.lower()

    if len(registers) > 20 or any(kw in text_lower for kw in
            ("microcontroller", "mcu", "cortex", "arm core", "flash memory",
             "stm32", "esp32", "nxp", "msp430", "renesas", "pic32")):
        return "mcu"

    if any(kw in text_lower for kw in
            ("accelerometer", "gyroscope", "temperature sensor", "pressure sensor",
             "humidity sensor", "magnetometer", "proximity", "light sensor",
             "current sensor", "adc converter")):
        return "sensor"

    if any(kw in text_lower for kw in
            ("motor driver", "h-bridge", "mosfet driver", "gate driver",
             "stepper", "pwm controller", "bldc driver")):
        return "driver"

    if any(kw in text_lower for kw in
            ("transceiver", "phy", "can controller", "usb controller",
             "ethernet controller", "wifi", "bluetooth", "lora",
             "rf transceiver", "bus interface")):
        return "interface"

    if any(kw in text_lower for kw in
            ("voltage regulator", "ldo", "buck converter", "boost converter",
             "power management", "pmic", "battery charger")):
        return "power"

    if any(kw in text_lower for kw in
            ("eeprom", "flash", "sram", "sdram", "nand", "nor",
             "memory controller", "ddr")):
        return "memory"

    return "component"


def _extract_constraints(text: str) -> list[str]:
    """Extract free-form hardware constraints from text.

    Looks for sentences containing constraint keywords like 'must', 'shall',
    'required', 'do not exceed', etc.
    """
    constraints: list[str] = []
    seen: set[str] = set()

    constraint_patterns = [
        re.compile(r"[^.]*\b(?:must not|must|shall not|shall|required to|do not exceed)\b[^.]*\.", re.IGNORECASE),
        re.compile(r"[^.]*\b(?:maximum (?:allowed|permitted)|minimum (?:required))\b[^.]*\.", re.IGNORECASE),
        re.compile(r"[^.]*\b(?:never|always|ensure|verify that|make sure)\b[^.]*\.", re.IGNORECASE),
        re.compile(r"[^.]*\b(?:caution|warning|danger|critical)[:\s][^.]*\.", re.IGNORECASE),
    ]

    for pat in constraint_patterns:
        for m in pat.finditer(text):
            constraint = m.group(0).strip()
            # Skip very short or very long matches
            if len(constraint) < 20 or len(constraint) > 500:
                continue
            # Normalize whitespace
            constraint = " ".join(constraint.split())
            if constraint not in seen:
                seen.add(constraint)
                constraints.append(constraint)

    return constraints[:50]  # Cap at 50 constraints
