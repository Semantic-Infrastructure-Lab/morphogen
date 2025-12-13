"""Physical unit system with dimensional analysis for Morphogen DSL.

This module provides a comprehensive unit system that supports:
- Seven SI base dimensions (mass, length, time, current, temperature, amount, luminosity)
- Dimensional analysis with unit algebra
- Unit compatibility checking
- Unit conversion between compatible units
- Parsing of unit expressions (e.g., "m/s", "kg*m/s^2")

Examples:
    >>> meter = Unit.meter()
    >>> second = Unit.second()
    >>> velocity = meter / second
    >>> velocity.dimensions
    Dimensions(length=1, time=-1)

    >>> Unit.parse("m/s") == velocity
    True

    >>> Unit.parse("kg*m/s^2").is_compatible_with(Unit.newton())
    True
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Union
import re
from fractions import Fraction


@dataclass(frozen=True)
class Dimensions:
    """Represents the dimensional formula of a physical quantity.

    Uses the seven SI base dimensions:
    - M: mass (kilogram)
    - L: length (meter)
    - T: time (second)
    - I: electric current (ampere)
    - Θ: thermodynamic temperature (kelvin)
    - N: amount of substance (mole)
    - J: luminous intensity (candela)

    Each dimension is represented as a rational number (Fraction) to support
    fractional exponents like sqrt(meter) = m^(1/2).
    """
    mass: Fraction = Fraction(0)        # M (kg)
    length: Fraction = Fraction(0)      # L (m)
    time: Fraction = Fraction(0)        # T (s)
    current: Fraction = Fraction(0)     # I (A)
    temperature: Fraction = Fraction(0) # Θ (K)
    amount: Fraction = Fraction(0)      # N (mol)
    luminosity: Fraction = Fraction(0)  # J (cd)

    def __mul__(self, other: 'Dimensions') -> 'Dimensions':
        """Multiply dimensions by adding exponents."""
        return Dimensions(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            current=self.current + other.current,
            temperature=self.temperature + other.temperature,
            amount=self.amount + other.amount,
            luminosity=self.luminosity + other.luminosity,
        )

    def __truediv__(self, other: 'Dimensions') -> 'Dimensions':
        """Divide dimensions by subtracting exponents."""
        return Dimensions(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            current=self.current - other.current,
            temperature=self.temperature - other.temperature,
            amount=self.amount - other.amount,
            luminosity=self.luminosity - other.luminosity,
        )

    def __pow__(self, exponent: Union[int, float, Fraction]) -> 'Dimensions':
        """Raise dimensions to a power by multiplying exponents."""
        exp = Fraction(exponent) if not isinstance(exponent, Fraction) else exponent
        return Dimensions(
            mass=self.mass * exp,
            length=self.length * exp,
            time=self.time * exp,
            current=self.current * exp,
            temperature=self.temperature * exp,
            amount=self.amount * exp,
            luminosity=self.luminosity * exp,
        )

    def is_dimensionless(self) -> bool:
        """Check if this represents a dimensionless quantity."""
        return all(d == 0 for d in [
            self.mass, self.length, self.time, self.current,
            self.temperature, self.amount, self.luminosity
        ])

    def __str__(self) -> str:
        """Human-readable representation of dimensions."""
        parts = []
        dim_names = [
            ('M', self.mass),
            ('L', self.length),
            ('T', self.time),
            ('I', self.current),
            ('Θ', self.temperature),
            ('N', self.amount),
            ('J', self.luminosity),
        ]
        for name, exp in dim_names:
            if exp != 0:
                if exp == 1:
                    parts.append(name)
                else:
                    parts.append(f"{name}^{exp}")
        return '·'.join(parts) if parts else '1'


@dataclass(frozen=True)
class Unit:
    """Represents a physical unit with dimensional analysis.

    A unit consists of:
    - name: Human-readable name (e.g., "meter", "second")
    - symbol: Short symbol (e.g., "m", "s")
    - dimensions: Dimensional formula
    - scale: Conversion factor to SI base units

    Units support algebraic operations:
    - Multiplication: meter * second -> meter·second
    - Division: meter / second -> meter/second (velocity)
    - Exponentiation: meter ** 2 -> square meter (area)
    """
    name: str
    symbol: str
    dimensions: Dimensions
    scale: float = 1.0  # Conversion factor to SI base units

    def __mul__(self, other: 'Unit') -> 'Unit':
        """Multiply two units."""
        # Handle dimensionless multiplication to avoid "1*m" symbols
        if self.symbol == "1":
            return other
        if other.symbol == "1":
            return self

        return Unit(
            name=f"{self.name}·{other.name}",
            symbol=f"{self.symbol}*{other.symbol}",
            dimensions=self.dimensions * other.dimensions,
            scale=self.scale * other.scale,
        )

    def __truediv__(self, other: 'Unit') -> 'Unit':
        """Divide two units."""
        return Unit(
            name=f"{self.name}/{other.name}",
            symbol=f"{self.symbol}/{other.symbol}",
            dimensions=self.dimensions / other.dimensions,
            scale=self.scale / other.scale,
        )

    def __pow__(self, exponent: Union[int, float]) -> 'Unit':
        """Raise unit to a power."""
        return Unit(
            name=f"{self.name}^{exponent}",
            symbol=f"{self.symbol}^{exponent}",
            dimensions=self.dimensions ** exponent,
            scale=self.scale ** exponent,
        )

    def is_compatible_with(self, other: 'Unit') -> bool:
        """Check if two units have the same dimensions (are convertible)."""
        return self.dimensions == other.dimensions

    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless unit."""
        return self.dimensions.is_dimensionless()

    def convert_to(self, other: 'Unit', value: float) -> float:
        """Convert a value from this unit to another compatible unit.

        Args:
            other: Target unit
            value: Value in this unit

        Returns:
            Value in target unit

        Raises:
            ValueError: If units are not compatible
        """
        if not self.is_compatible_with(other):
            raise ValueError(
                f"Cannot convert {self.symbol} to {other.symbol}: "
                f"incompatible dimensions {self.dimensions} vs {other.dimensions}"
            )
        return value * (self.scale / other.scale)

    @staticmethod
    def dimensionless() -> 'Unit':
        """Create a dimensionless unit."""
        return Unit("dimensionless", "1", Dimensions())

    # SI Base Units

    @staticmethod
    def meter() -> 'Unit':
        """SI unit of length."""
        return Unit("meter", "m", Dimensions(length=Fraction(1)))

    @staticmethod
    def kilogram() -> 'Unit':
        """SI unit of mass."""
        return Unit("kilogram", "kg", Dimensions(mass=Fraction(1)))

    @staticmethod
    def second() -> 'Unit':
        """SI unit of time."""
        return Unit("second", "s", Dimensions(time=Fraction(1)))

    @staticmethod
    def ampere() -> 'Unit':
        """SI unit of electric current."""
        return Unit("ampere", "A", Dimensions(current=Fraction(1)))

    @staticmethod
    def kelvin() -> 'Unit':
        """SI unit of temperature."""
        return Unit("kelvin", "K", Dimensions(temperature=Fraction(1)))

    @staticmethod
    def mole() -> 'Unit':
        """SI unit of amount of substance."""
        return Unit("mole", "mol", Dimensions(amount=Fraction(1)))

    @staticmethod
    def candela() -> 'Unit':
        """SI unit of luminous intensity."""
        return Unit("candela", "cd", Dimensions(luminosity=Fraction(1)))

    # Common Derived Units

    @staticmethod
    def newton() -> 'Unit':
        """SI unit of force: kg·m/s²."""
        return Unit(
            "newton", "N",
            Dimensions(mass=Fraction(1), length=Fraction(1), time=Fraction(-2))
        )

    @staticmethod
    def joule() -> 'Unit':
        """SI unit of energy: kg·m²/s²."""
        return Unit(
            "joule", "J",
            Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-2))
        )

    @staticmethod
    def watt() -> 'Unit':
        """SI unit of power: kg·m²/s³."""
        return Unit(
            "watt", "W",
            Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-3))
        )

    @staticmethod
    def pascal() -> 'Unit':
        """SI unit of pressure: kg/(m·s²)."""
        return Unit(
            "pascal", "Pa",
            Dimensions(mass=Fraction(1), length=Fraction(-1), time=Fraction(-2))
        )

    @staticmethod
    def hertz() -> 'Unit':
        """SI unit of frequency: 1/s."""
        return Unit(
            "hertz", "Hz",
            Dimensions(time=Fraction(-1))
        )

    @staticmethod
    def volt() -> 'Unit':
        """SI unit of voltage: kg·m²/(A·s³)."""
        return Unit(
            "volt", "V",
            Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-3), current=Fraction(-1))
        )

    @staticmethod
    def coulomb() -> 'Unit':
        """SI unit of electric charge: A·s."""
        return Unit(
            "coulomb", "C",
            Dimensions(current=Fraction(1), time=Fraction(1))
        )

    @staticmethod
    def ohm() -> 'Unit':
        """SI unit of resistance: kg·m²/(A²·s³)."""
        return Unit(
            "ohm", "Ω",
            Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-3), current=Fraction(-2))
        )

    # Common prefixed units

    @staticmethod
    def centimeter() -> 'Unit':
        """Centimeter: 0.01 m."""
        return Unit("centimeter", "cm", Dimensions(length=Fraction(1)), scale=0.01)

    @staticmethod
    def kilometer() -> 'Unit':
        """Kilometer: 1000 m."""
        return Unit("kilometer", "km", Dimensions(length=Fraction(1)), scale=1000.0)

    @staticmethod
    def gram() -> 'Unit':
        """Gram: 0.001 kg."""
        return Unit("gram", "g", Dimensions(mass=Fraction(1)), scale=0.001)

    @staticmethod
    def millisecond() -> 'Unit':
        """Millisecond: 0.001 s."""
        return Unit("millisecond", "ms", Dimensions(time=Fraction(1)), scale=0.001)


# Unit registry for parsing
_UNIT_REGISTRY: Dict[str, Unit] = {
    # SI base units
    "m": Unit.meter(),
    "kg": Unit.kilogram(),
    "s": Unit.second(),
    "A": Unit.ampere(),
    "K": Unit.kelvin(),
    "mol": Unit.mole(),
    "cd": Unit.candela(),

    # Common derived units
    "N": Unit.newton(),
    "J": Unit.joule(),
    "W": Unit.watt(),
    "Pa": Unit.pascal(),
    "Hz": Unit.hertz(),
    "V": Unit.volt(),
    "C": Unit.coulomb(),
    "Ω": Unit.ohm(),

    # Prefixed units
    "cm": Unit.centimeter(),
    "km": Unit.kilometer(),
    "g": Unit.gram(),
    "ms": Unit.millisecond(),

    # Dimensionless
    "1": Unit.dimensionless(),
}


def parse_unit(unit_str: str) -> Unit:
    """Parse a unit expression string into a Unit object.

    Supports:
    - Simple units: "m", "kg", "s"
    - Products: "kg*m", "m*s"
    - Quotients: "m/s", "kg/m^3"
    - Powers: "m^2", "s^-1"
    - Combinations: "kg*m/s^2" (force)

    Args:
        unit_str: Unit expression string

    Returns:
        Parsed Unit object

    Raises:
        ValueError: If the unit expression is invalid

    Examples:
        >>> parse_unit("m")
        Unit(name='meter', symbol='m', ...)

        >>> parse_unit("m/s")
        Unit(name='meter/second', symbol='m/s', ...)

        >>> parse_unit("kg*m/s^2")  # Force (Newton)
        Unit(name='kilogram·meter/second^2', ...)
    """
    # Handle empty or whitespace
    unit_str = unit_str.strip()
    if not unit_str or unit_str == "1":
        return Unit.dimensionless()

    # First, split by division to handle numerator/denominator
    parts = unit_str.split('/')
    if len(parts) > 2:
        raise ValueError(f"Invalid unit expression: multiple divisions not supported in '{unit_str}'")

    numerator = parts[0]
    denominator = parts[1] if len(parts) == 2 else None

    # Parse numerator (products and powers)
    num_unit = _parse_product(numerator)

    # Parse denominator if present
    if denominator:
        denom_unit = _parse_product(denominator)
        return num_unit / denom_unit

    return num_unit


def _parse_product(expr: str) -> Unit:
    """Parse a product expression like 'kg*m' or 'm^2'."""
    # Split by multiplication
    terms = expr.split('*')

    result = Unit.dimensionless()
    for term in terms:
        term = term.strip()
        if not term:
            continue

        # Check for power (e.g., "m^2")
        if '^' in term:
            base, exp_str = term.split('^', 1)
            base = base.strip()
            try:
                exponent = int(exp_str.strip())
            except ValueError:
                raise ValueError(f"Invalid exponent in '{term}'")

            if base not in _UNIT_REGISTRY:
                raise ValueError(f"Unknown unit '{base}' in '{term}'")

            result = result * (_UNIT_REGISTRY[base] ** exponent)
        else:
            # Simple unit
            if term not in _UNIT_REGISTRY:
                raise ValueError(f"Unknown unit '{term}'")
            result = result * _UNIT_REGISTRY[term]

    return result


def check_unit_compatibility(unit1_str: Optional[str], unit2_str: Optional[str]) -> bool:
    """Check if two unit strings are dimensionally compatible.

    Args:
        unit1_str: First unit string (can be None)
        unit2_str: Second unit string (can be None)

    Returns:
        True if compatible, False otherwise

    Note:
        - None is compatible with any unit
        - Empty string is treated as dimensionless
    """
    # None matches anything (for backward compatibility)
    if unit1_str is None or unit2_str is None:
        return True

    # Both empty/whitespace means dimensionless
    if not unit1_str.strip() and not unit2_str.strip():
        return True

    try:
        unit1 = parse_unit(unit1_str)
        unit2 = parse_unit(unit2_str)
        return unit1.is_compatible_with(unit2)
    except ValueError:
        # If parsing fails, fall back to string comparison
        return unit1_str == unit2_str


def get_unit_dimensions(unit_str: Optional[str]) -> Optional[Dimensions]:
    """Get the dimensional formula for a unit string.

    Args:
        unit_str: Unit string to analyze

    Returns:
        Dimensions object, or None if unit_str is None or invalid
    """
    if unit_str is None:
        return None

    try:
        unit = parse_unit(unit_str)
        return unit.dimensions
    except ValueError:
        return None
