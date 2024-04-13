from typing import (
    List,
    Dict,
    overload,
    final,
)
from numpy import ndarray


class NgSpiceCommandError(RuntimeError):
    """
    Raised when an error occurs while executing a command in Ngspice.
    """
    pass


class Shared:
    """
    A shared object for interacting with Ngspice.

    Attributes:
        ngspice_id (int): The identifier of the Ngspice instance.
    """

    ngspice_id: int

    @classmethod
    def new_instance(cls, ngspice_id: int = 0, send_data: bool = False, verbose: bool = False) -> Shared:
        """
        Create a new instance of the Ngspice shared object.

        :param ngspice_id: The id of the Ngspice instance. Defaults to 0.
        :param send_data: Whether to send data. Defaults to False.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A new instance of the Ngspice shared object.
        """
    def __init__(self, ngspice_id: int = 0, send_data: bool = False, verbose: bool = False):
        """
        Initialize a Shared instance.

        :param ngspice_id: The id of the Ngspice instance. Defaults to 0.
        :param send_data: Whether to send data. Defaults to False.
        :param verbose: Whether to print verbose output. Defaults to False.
        """
    def _init_ngspice(self, send_data: bool = False) -> None:
        """
        Initialize Ngspice on the instance.

        :param send_data: Whether to send data. Defaults to False.
        """
    @final
    def clear_output(self) -> None:
        """
        Clear the output (stdout and stderr) of the Shared instance.
        """
    @property
    def stdout(self) -> str:
        """
        Get the standard output of the Shared instance.

        :return: The joined standard output string.
        """
    @property
    def stderr(self) -> str:
        """
        Get the standard error of the Shared instance.

        :return: The joined standard error string.
        """
    @property
    def _stdout(self) -> List[str]:
        """
        Get the standard output of the Shared instance.

        :return: The list of standard output lines.
        """
    @property
    def _stderr(self) -> List[str]:
        """
        Get the standard error of the Shared instance.

        :return: The list of standard error lines.
        """
    @property
    def plot_names(self) -> List[str]:
        """
        Get the names of the plots in the Shared instance.

        :return: The list of plot names.
        """
    @overload
    def exec_command(self, command: str, join_lines: bool = True) -> str: ...
    @overload
    def exec_command(self, command: str, join_lines: bool = False) -> List[str]: ...
    def exec_command(self, command: str, join_lines: bool = True) -> str | List[str]:
        """
        Execute a command in Ngspice.

        :param command: The command to execute.
        :param join_lines: Whether to join the output lines. Defaults to True.
        :return: The standard output of the command.
        """
    def load_circuit(self, circuit: str) -> None:
        """
        Load a circuit into Ngspice.

        :param circuit: The circuit to load.
        """
    def run(self, background: bool = False) -> None:
        """
        Run the simulation in Ngspice.

        :param background: Whether to run the simulation in the background. Defaults to False.
        """
    def plot(self, plot_name: str) -> Dict[str, ndarray]: ...

    def send_char(self, message: str, ngspice_id: int) -> int: ...
    def send_stat(self, message: str, ngspice_id: int) -> int: ...
    def send_data(self, actual_vector_values: Dict[str, complex], number_of_vectors: int, ngspice_id: int) -> int: ...
    # TODO: def send_init_data

    def get_vsrc_data(self, voltage: float, time: float, node_name: str, ngspice_id: int) -> int: ...
    def get_isrc_data(self, current: float, time: float, node_name: str, ngspice_id: int) -> int: ...
    def get_sync_data(self, actual_time: float, delta_time: float, old_delta_time: float, redostep: int, ngspice_id: int, loc: int) -> int: ...
