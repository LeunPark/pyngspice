try:
    from PySpice.Spice.NgSpice.Simulation import NgSpiceCircuitSimulator
    from PySpice.Spice.NgSpice.Shared import (
        Vector as _Vector,
        Plot as _Plot,
    )
    from PySpice.Spice.Netlist import Circuit as _Circuit
except ImportError:
    raise ImportError("PySpice is not installed. Run `pip install PySpice`.")

from . import Shared


class Vector(_Vector):
    @property
    def is_voltage_node(self):
        return self._type == "voltage" and not self.is_interval_parameter

    @property
    def is_branch_current(self):
        return self._type == "current" and not self.is_interval_parameter


class Plot(_Plot):
    ...


class NgSpiceSharedCircuitSimulator(NgSpiceCircuitSimulator):
    def __init__(self, circuit, **kwargs):
        super().__init__(circuit, pipe=False, **kwargs)
        self._ngspice_shared: Shared = kwargs.get('ngspice_shared') or Shared.new_instance()

    @property
    def ngspice(self):
        return self._ngspice_shared

    def _run(self, analysis_method, *args, **kwargs):
        super()._run(analysis_method, *args, **kwargs)

        self._ngspice_shared.destroy()
        self._ngspice_shared.load_circuit(str(self))
        self._ngspice_shared.run()
        # self._logger.debug(str(self._ngspice_shared.plot_names))
        self.reset_analysis()

        plot_name = self._ngspice_shared.last_plot
        if plot_name == 'const':
            raise NameError('Simulation failed')
        return self._ngspice_shared.pyspice_plot(self, plot_name).to_analysis()


class Circuit(_Circuit):
    def simulator(self, *args, **kwargs):
        return NgSpiceSharedCircuitSimulator(self, *args, **kwargs)
