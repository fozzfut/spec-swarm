"""Expert profile loading and suggestion for hardware spec analysis."""

from __future__ import annotations

from pathlib import Path

import yaml

_BUILTIN_DIR = Path(__file__).parent / "experts"


class ExpertProfiler:
    """Load and suggest expert profiles for hardware specification analysis."""

    def __init__(self, custom_dirs: list[Path] | None = None):
        self._custom_dirs = custom_dirs or []

    def list_profiles(self) -> list[dict]:
        """List all available expert profiles."""
        profiles = []
        for yaml_file in sorted(_BUILTIN_DIR.glob("*.yaml")):
            profiles.append(self._load_yaml(yaml_file))
        for d in self._custom_dirs:
            if d.exists():
                for yaml_file in sorted(d.glob("*.yaml")):
                    profiles.append(self._load_yaml(yaml_file))
        return profiles

    def load_profile(self, name: str) -> dict:
        """Load a specific expert profile by name (filename without .yaml)."""
        path = _BUILTIN_DIR / f"{name}.yaml"
        if path.exists():
            return self._load_yaml(path)
        for d in self._custom_dirs:
            path = d / f"{name}.yaml"
            if path.exists():
                return self._load_yaml(path)
        raise FileNotFoundError(f"Expert profile '{name}' not found")

    def suggest_experts(
        self,
        specs: list[dict],
        protocols_used: list[str] | None = None,
        categories_used: list[str] | None = None,
    ) -> list[dict]:
        """Suggest expert profiles based on components, protocols, and categories found.

        Args:
            specs: List of HardwareSpec.to_dict() results.
            protocols_used: List of protocol names (SPI, I2C, etc.).
            categories_used: List of component categories (mcu, sensor, etc.).

        Returns:
            Sorted list of expert suggestions with confidence scores.
        """
        if protocols_used is None:
            protocols_used = []
        if categories_used is None:
            categories_used = []

        # Collect all relevant keywords from specs
        all_keywords: set[str] = set()

        for spec in specs:
            # Component category
            cat = spec.get("category", "")
            if cat:
                all_keywords.add(cat.lower())

            # Protocols
            for proto in spec.get("protocols", []):
                p = proto.get("protocol", "")
                if p:
                    all_keywords.add(p.lower())

            # Peripheral types from registers
            for reg in spec.get("registers", []):
                name = reg.get("name", "").upper()
                for ptype in ("GPIO", "UART", "SPI", "I2C", "CAN", "USB",
                              "ADC", "DAC", "TIMER", "PWM", "DMA", "WDG",
                              "RTC", "ETH", "SDIO"):
                    if ptype in name:
                        all_keywords.add(ptype.lower())

            # Tags
            for tag in spec.get("tags", []):
                all_keywords.add(tag.lower())

            # Timing constraints
            if spec.get("timing"):
                all_keywords.add("timing")

            # Power specs
            if spec.get("power"):
                all_keywords.add("power")

            # Memory map
            if spec.get("memory_map"):
                all_keywords.add("memory")

            # Safety-related constraints
            for constraint in spec.get("constraints", []):
                constraint_lower = constraint.lower()
                if any(kw in constraint_lower for kw in
                       ("safety", "redundan", "watchdog", "fail-safe", "critical",
                        "iec", "misra", "protect")):
                    all_keywords.add("safety")

        # Add explicit protocol/category hints
        for p in protocols_used:
            all_keywords.add(p.lower())
        for c in categories_used:
            all_keywords.add(c.lower())

        # Score each profile
        suggestions = []
        for profile in self.list_profiles():
            score = self._score_profile(profile, all_keywords)
            if score > 0:
                profile_name = Path(profile.get("_source_file", "")).stem
                suggestions.append({
                    "profile_name": profile_name,
                    "name": profile.get("name", ""),
                    "description": profile.get("description", ""),
                    "confidence": min(score, 1.0),
                })

        suggestions.sort(key=lambda s: s["confidence"], reverse=True)
        return suggestions

    def _score_profile(self, profile: dict, keywords: set[str]) -> float:
        """Score a profile's relevance based on keywords found in specs."""
        score = 0.0
        relevance = profile.get("relevance_keywords", [])

        for kw in relevance:
            if kw.lower() in keywords:
                score += 0.25

        # Boost based on profile name matching keywords
        profile_name = profile.get("name", "").lower()
        for kw in keywords:
            if kw in profile_name:
                score += 0.15

        return score

    def _load_yaml(self, path: Path) -> dict:
        """Load and validate a YAML profile."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            return {
                "name": path.stem,
                "description": f"Error loading: {exc}",
                "_source_file": str(path),
            }
        if not isinstance(data, dict):
            return {
                "name": path.stem,
                "description": "Invalid format",
                "_source_file": str(path),
            }
        data["_source_file"] = str(path)
        return data
