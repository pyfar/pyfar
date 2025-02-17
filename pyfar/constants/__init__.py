"""Module for constants and physical properties of air."""


from .speed_of_sound import (
    speed_of_sound_simple,
)

from .medium_attenuation import (
    air_attenuation,
)

__all__ = [
    'speed_of_sound_simple',
    'air_attenuation',
]
