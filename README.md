# Open_Quantum_Systems_RI
## System: spin
### T = 0
  - *single_oscill_cont_trott.py*: classes to simulate a single oscillator exactly either with Linblad equation or with Repeated interaction protocol with (or without) trotterization. Also it can be compared with the results from a quantum circuit.
  - *qiskit_nonoise.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, without noise.
#### Add noise:
  - *qiskit_flipnoise.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, with noise. We consider flips either when measuring, when reseting a qubit or when a gate is applied
  - *qiskit_backends.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, with the noise of an IBM simulator (backend)
  - *qiskit_depolarizing.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, with noise. We consider depolarizing in the gates.
  - *qiskit_readout.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, with noise. We consider the probability of error when measuring.
  - *qiskit_relaxation.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI, with noise. We consider decoherence by thermal relaxation.
### Finite temperature
  - *qiskit_thermalstate.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI. The ancilla is in a thermal Gibbs state. To obtain it we have an extra qubit and we apply an ry gate to the ancilla and then a not gate to the ancilla and extra qubit. Can be calculated noiseless or with backend.
  - *qiskit_thermalstate2.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI. The ancilla is in a thermal Gibbs state. To obtain it we do classical sampling. Can be calculated noiseless or with backend.
  - *qiskit_bosonicbath.ipynb*: jupyer notebook to do the quantum circuit for a single oscillator under RI. The bath is bosonic, and it is obtained by adding a thermal dependance related with bosonic statistics in the coupling g.

## System: spin / quantum harmonic oscillator
### Harmonic Oscillator coupled to a spin. Spin coupled to a bath.
- *qiskit_harmonicoscill.ipynb*: Quantum circuit of harmonic oscillator coupled to a spin (transversal or longitudinal). The spin is coupled to a bath, treated as an ancilla RI model. The bath can be at a finite temperature and be either a boson or spin bath.
- *linblad_harmtransspinbosonicbath.py*: harmonic oscillator coupled to a spin (transversal coupling). The spin is coupled to a bath that can be at a finite temperature and be either a boson or spin bath. The code solves lindblad equation for such system.
### Harmonic Oscillator coupled to a bath.
- *linblad_harmbath.py*: harmonic oscillator coupled to a bosonic bath that can be at a finite temperature. The code solves lindblad equation for such system.
- *qiskit_harmbath.ipynb*: Quantum circuit for harmonic oscillator coupled to a bath (boson or spin) at finite temperature. The bath is treated as an ancilla in the RI model.
### Spin coupled to a Harmonic Oscillator. Harmonic Oscillator coupled to a bath
- *linblad_spinharmtransbosonicbath.py*: Spin coupled to a harmonic oscillator and the harmonic oscillator is coupled to a boson bath that can be at a finite temperature. The code solves lindblad equation for such system.

