"""Module for constants and physical properties of air."""


from .speed_of_sound import (
    speed_of_sound_simple,
)

from .air_attenuation import (
    air_attenuation_iso,
)

__all__ = [
    'speed_of_sound_simple',
    'air_attenuation_iso',
]
