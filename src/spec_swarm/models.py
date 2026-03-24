"""Data models for hardware specification analysis."""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SpecType(str, Enum):
    DATASHEET = "datasheet"
    REFERENCE_MANUAL = "reference_manual"
    APPLICATION_NOTE = "application_note"
    PROTOCOL_SPEC = "protocol_spec"
    REQUIREMENTS = "requirements"
    SCHEMATIC = "schematic"
    PINOUT = "pinout"


class PeripheralType(str, Enum):
    GPIO = "gpio"
    UART = "uart"
    SPI = "spi"
    I2C = "i2c"
    CAN = "can"
    USB = "usb"
    ADC = "adc"
    DAC = "dac"
    TIMER = "timer"
    PWM = "pwm"
    DMA = "dma"
    WATCHDOG = "watchdog"
    RTC = "rtc"
    ETHERNET = "ethernet"
    SDIO = "sdio"
    MODBUS = "modbus"
    CANOPEN = "canopen"
    ETHERCAT = "ethercat"
    PROFINET = "profinet"
    OPCUA = "opcua"
    IOLINK = "iolink"
    OTHER = "other"


@dataclass
class Register:
    """A hardware register definition."""
    name: str = ""
    address: str = ""  # hex address like "0x40021000"
    size_bits: int = 32
    reset_value: str = "0x00000000"
    access: str = "rw"  # rw, ro, wo, rc_w1, etc.
    description: str = ""
    fields: list[dict] = field(default_factory=list)  # [{name, bits, description, values}]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "address": self.address,
            "size_bits": self.size_bits,
            "reset_value": self.reset_value,
            "access": self.access,
            "description": self.description,
            "fields": list(self.fields),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Register":
        return cls(
            name=d.get("name", ""),
            address=d.get("address", ""),
            size_bits=d.get("size_bits", 32),
            reset_value=d.get("reset_value", "0x00000000"),
            access=d.get("access", "rw"),
            description=d.get("description", ""),
            fields=d.get("fields", []),
        )


@dataclass
class PinConfig:
    """MCU pin configuration."""
    pin: str = ""         # "PA0", "PB5", etc.
    function: str = ""    # "GPIO", "UART1_TX", "SPI1_MOSI"
    af_number: int = -1   # Alternate function number
    direction: str = ""   # "input", "output", "analog", "alternate"
    pull: str = ""        # "none", "up", "down"
    speed: str = ""       # "low", "medium", "high", "very_high"
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v != "" and v != -1}

    @classmethod
    def from_dict(cls, d: dict) -> "PinConfig":
        return cls(
            pin=d.get("pin", ""),
            function=d.get("function", ""),
            af_number=d.get("af_number", -1),
            direction=d.get("direction", ""),
            pull=d.get("pull", ""),
            speed=d.get("speed", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class ProtocolConfig:
    """Communication protocol configuration."""
    protocol: str = ""     # "SPI", "I2C", "UART", "CAN", etc.
    instance: str = ""     # "SPI1", "I2C2", etc.
    role: str = ""         # "master", "slave"
    speed: str = ""        # "100kHz", "1MHz", "115200 baud"
    mode: str = ""         # protocol-specific (SPI mode 0-3, etc.)
    data_bits: int = 8
    word_order: str = ""   # "MSB first", "LSB first"
    pins: list[PinConfig] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v != "" and k != "pins"}
        if self.pins:
            d["pins"] = [p.to_dict() for p in self.pins]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolConfig":
        pins_data = d.get("pins", [])
        pins = [PinConfig.from_dict(p) if isinstance(p, dict) else p for p in pins_data]
        return cls(
            protocol=d.get("protocol", ""),
            instance=d.get("instance", ""),
            role=d.get("role", ""),
            speed=d.get("speed", ""),
            mode=d.get("mode", ""),
            data_bits=d.get("data_bits", 8),
            word_order=d.get("word_order", ""),
            pins=pins,
            notes=d.get("notes", ""),
        )


@dataclass
class TimingConstraint:
    """A timing requirement from a datasheet."""
    parameter: str = ""      # "t_startup", "t_conv", "SPI clock max"
    min_value: str = ""      # "10 us"
    typ_value: str = ""      # "50 us"
    max_value: str = ""      # "100 us"
    unit: str = ""           # "us", "ms", "ns", "MHz"
    condition: str = ""      # "VDD = 3.3V, T = 25C"
    source: str = ""         # "Datasheet Table 47"
    critical: bool = False   # violation causes hardware damage or malfunction

    def to_dict(self) -> dict:
        d: dict = {}
        for k, v in self.__dict__.items():
            if k == "critical":
                d[k] = v
            elif v != "":
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TimingConstraint":
        return cls(
            parameter=d.get("parameter", ""),
            min_value=d.get("min_value", ""),
            typ_value=d.get("typ_value", ""),
            max_value=d.get("max_value", ""),
            unit=d.get("unit", ""),
            condition=d.get("condition", ""),
            source=d.get("source", ""),
            critical=d.get("critical", False),
        )


@dataclass
class PowerSpec:
    """Power supply specification."""
    rail: str = ""           # "VDD", "VDDA", "VBAT"
    min_voltage: str = ""    # "2.7V"
    typ_voltage: str = ""    # "3.3V"
    max_voltage: str = ""    # "3.6V"
    max_current: str = ""    # "150mA"
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, d: dict) -> "PowerSpec":
        return cls(
            rail=d.get("rail", ""),
            min_voltage=d.get("min_voltage", ""),
            typ_voltage=d.get("typ_voltage", ""),
            max_voltage=d.get("max_voltage", ""),
            max_current=d.get("max_current", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class MemoryRegion:
    """Memory map region."""
    name: str = ""           # "Flash", "SRAM1", "Peripheral"
    start_address: str = ""  # "0x08000000"
    end_address: str = ""    # "0x0807FFFF"
    size: str = ""           # "512 KB"
    access: str = ""         # "rx" (read-execute), "rw", "ro"
    description: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRegion":
        return cls(
            name=d.get("name", ""),
            start_address=d.get("start_address", ""),
            end_address=d.get("end_address", ""),
            size=d.get("size", ""),
            access=d.get("access", ""),
            description=d.get("description", ""),
        )


@dataclass
class HardwareSpec:
    """Complete hardware specification for a component or system."""
    id: str = ""
    name: str = ""           # "STM32F407VG", "BME280", "MCP2515"
    category: str = ""       # "mcu", "sensor", "driver", "interface"
    manufacturer: str = ""
    part_number: str = ""
    source_doc: str = ""     # path to original document
    spec_type: SpecType = SpecType.DATASHEET

    registers: list[Register] = field(default_factory=list)
    pins: list[PinConfig] = field(default_factory=list)
    protocols: list[ProtocolConfig] = field(default_factory=list)
    timing: list[TimingConstraint] = field(default_factory=list)
    power: list[PowerSpec] = field(default_factory=list)
    memory_map: list[MemoryRegion] = field(default_factory=list)

    constraints: list[str] = field(default_factory=list)  # free-form constraints
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = "hw-" + secrets.token_hex(4)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "manufacturer": self.manufacturer,
            "part_number": self.part_number,
            "source_doc": self.source_doc,
            "spec_type": self.spec_type.value if isinstance(self.spec_type, SpecType) else self.spec_type,
            "registers": [r.to_dict() for r in self.registers],
            "pins": [p.to_dict() for p in self.pins],
            "protocols": [p.to_dict() for p in self.protocols],
            "timing": [t.to_dict() for t in self.timing],
            "power": [p.to_dict() for p in self.power],
            "memory_map": [m.to_dict() for m in self.memory_map],
            "constraints": list(self.constraints),
            "notes": list(self.notes),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareSpec":
        spec_type_raw = d.get("spec_type", "datasheet")
        try:
            spec_type = SpecType(spec_type_raw)
        except ValueError:
            spec_type = SpecType.DATASHEET

        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            category=d.get("category", ""),
            manufacturer=d.get("manufacturer", ""),
            part_number=d.get("part_number", ""),
            source_doc=d.get("source_doc", ""),
            spec_type=spec_type,
            registers=[Register.from_dict(r) for r in d.get("registers", [])],
            pins=[PinConfig.from_dict(p) for p in d.get("pins", [])],
            protocols=[ProtocolConfig.from_dict(p) for p in d.get("protocols", [])],
            timing=[TimingConstraint.from_dict(t) for t in d.get("timing", [])],
            power=[PowerSpec.from_dict(p) for p in d.get("power", [])],
            memory_map=[MemoryRegion.from_dict(m) for m in d.get("memory_map", [])],
            constraints=d.get("constraints", []),
            notes=d.get("notes", []),
            tags=d.get("tags", []),
        )


@dataclass
class SpecSession:
    """A spec analysis session."""
    id: str = ""
    project_path: str = ""
    created_at: str = ""
    specs: list[HardwareSpec] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)  # constraint violations, missing info

    def __post_init__(self):
        if not self.id:
            self.id = "spec-" + secrets.token_hex(4)
        if not self.created_at:
            self.created_at = now_iso()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_path": self.project_path,
            "created_at": self.created_at,
            "specs": [s.to_dict() for s in self.specs],
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecSession":
        return cls(
            id=d.get("id", ""),
            project_path=d.get("project_path", ""),
            created_at=d.get("created_at", ""),
            specs=[HardwareSpec.from_dict(s) for s in d.get("specs", [])],
            findings=d.get("findings", []),
        )
