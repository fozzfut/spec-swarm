# SpecSwarm

Hardware specification analyzer for embedded software development. Part of the Swarm Suite.

SpecSwarm parses datasheets, reference manuals, and hardware documentation to extract structured data -- register maps, pin configurations, protocol parameters, timing constraints, power specifications, and memory layouts. It makes this information available to AI agents via MCP tools before any code exists.

## Installation

```bash
pip install spec-swarm-ai
```

For PDF datasheet support:

```bash
pip install spec-swarm-ai[pdf]
```

## Quick Start

### As MCP Server

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "spec-swarm": {
      "command": "spec-swarm",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### CLI Usage

```bash
# Start the MCP server
spec-swarm serve --transport stdio

# Ingest a datasheet
spec-swarm ingest path/to/STM32F407_datasheet.pdf --component STM32F407VG

# Check session status
spec-swarm status
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `spec_start_session` | Start a spec analysis session for a project |
| `spec_list_sessions` | List all spec analysis sessions |
| `spec_ingest` | Ingest a document and extract hardware specs |
| `spec_add_manual` | Manually add a hardware specification |
| `spec_get_registers` | Get register map for a component |
| `spec_get_pins` | Get pin configuration |
| `spec_get_protocols` | Get communication protocol configurations |
| `spec_get_timing` | Get timing constraints |
| `spec_get_memory_map` | Get memory map regions |
| `spec_get_constraints` | Get all hardware constraints |
| `spec_search` | Search specs by keyword |
| `spec_check_conflicts` | Check for pin collisions, bus conflicts, power budget violations |
| `spec_suggest_experts` | Suggest spec expert profiles for the project |
| `spec_export_for_arch` | Export specs as architectural constraints for ArchSwarm |
| `spec_get_summary` | Get summary of all specs in a session |

## Supported Document Formats

- **PDF** (requires `pymupdf` -- install with `pip install spec-swarm-ai[pdf]`)
- **Plain text** (.txt)
- **Markdown** (.md)
- **reStructuredText** (.rst)
- **CSV** (.csv)

## Expert Profiles

SpecSwarm includes 8 expert profiles for hardware specification analysis:

- **MCU Peripherals** -- GPIO, clock tree, interrupts, DMA
- **Communication Protocols** -- SPI, I2C, UART, CAN, USB, Ethernet
- **Power Management** -- voltage rails, current budgets, sleep modes
- **Sensor Interfaces** -- ADC, sampling, calibration, filtering
- **Motor Control** -- PWM, H-bridge, encoders, FOC
- **Memory Layout** -- flash/RAM partitioning, linker scripts, bootloader
- **Timing Constraints** -- deadlines, watchdog, jitter, clock drift
- **Safety Requirements** -- IEC 61508, redundancy, fail-safe design

## Integration with Swarm Suite

SpecSwarm stores extracted specifications in swarm-kb, making them available to:

- **ArchSwarm** -- hardware constraints inform architecture decisions
- **ReviewSwarm** -- verify code against datasheet requirements
- **FixSwarm** -- apply hardware-aware fixes

## License

MIT
