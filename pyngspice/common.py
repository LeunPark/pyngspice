try:
    # from PySpice.Spice.NgSpice.Simulation import NgSpiceCircuitSimulator
    from PySpice.Spice.NgSpice.Shared import Vector
except ImportError:
    raise ImportError("PySpice is not installed. Run `pip install PySpice`.")


class _Vector(Vector):
    @property
    def is_voltage_node(self):
        return self._type == "voltage" and not self.is_interval_parameter

    @property
    def is_branch_current(self):
        return self._type == "current" and not self.is_interval_parameter
