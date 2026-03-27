from dataclasses import dataclass


@dataclass
class MemoryRegion:
    label: str
    bytes_reserved: int
