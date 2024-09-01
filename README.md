# pyngspice
pyngspice is a Python binding for the circuit simulator [Ngspice](https://ngspice.sourceforge.io/), enabling seamless integration of Ngspice simulations within Python scripts. 
This project draws significant inspiration from PySpice but aims for faster performance by leveraging Python’s C extension capabilities.

## Installation

### Step 1: Install Ngspice
Before installing, you need to set up Ngspice on your system.
#### Windows
1. Download the latest Ngspice DLL package from [this link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/42/ngspice-42_dll_64.7z).
2. Extract the downloaded file and rename the `Spice64_dll` folder to `Spice64`.
3. Move the `Spice64` folder to the `C:\\` directory.

#### Linux

**For Debian-based distributions (e.g., Ubuntu):**
```bash
sudo apt install libngspice0-dev ngspice
```

**For Fedora-based distributions:**
```bash
sudo dnf install libngspice-devel ngspice-codemodel
```
*(Note: The Fedora-based installation has not yet been tested.)*

#### macOS
For macOS users, Ngspice can be installed using Homebrew:
```bash
brew install ngspice
```

#### Multithreading version (Optional)
For those requiring multithreading support of Ngspice, you can install the OpenMP version using the following script (supported on Linux and macOS):
```bash
curl -sSL https://raw.githubusercontent.com/LeunPark/pyngspice/main/scripts/install-ngspice-omp.sh | bash -
```

### Step 2: Install pyngspice
Once Ngspice is installed, you can proceed to install `pyngspice`:
```bash
pip install git+https://github.com/LeunPark/pyngspice.git
```
If you encounter errors related to `pkg-config` during installation, you may need to install it separately.

## Usage
```python
import pyngspice
```
