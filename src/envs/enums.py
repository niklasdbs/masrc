"""
Module contains custom enums used by the environment.
"""

#do not use the enum classes for performance

class ParkingStatus:#(IntEnum):
    """Describes the status of the parking space."""
    FREE = 0
    OCCUPIED = 1
    IN_VIOLATION = 2
    FINED = 3
    UNKNOWN = 4


class EventType:#(Enum):
    """Describes the type of the agent event."""
    ARRIVAL = 0
    DEPARTURE = 1
    VIOLATION = 2
