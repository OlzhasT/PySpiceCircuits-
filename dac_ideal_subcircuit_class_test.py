import PySpice.Logging.Logging as Logging
#logger = Logging.setup_logging()
from PySpice.Unit.Units import *
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Netlist import DeviceModel
from PySpice.Spice.Netlist import SubCircuitFactory
import matplotlib.pyplot as plt
import numpy as np

'''
BitLogic Subcircuit Schematic

    vdd
     |
bx--\----trip
     |
     |---bxl
     |
bx----/--trip
     |
    vss

Trip V generation Schematic

    vdd
     |
   |res|
     |---trip
   |res|
     |
    vss

'''
class BitLogic(SubCircuitFactory):
    __name__='BITLOGIC' # subcircuit name
    __nodes__=('bx', 'bxl', 'vdd', 'vss') # subcircuit nodes set
    def __init__(self):
        super().__init__() # initialize a parent class - SubCircuitFactory
        # add model parameters for the ideal switches to the BitLogic Subcircuit
        self.swmod_params = {'ron':0.1, 'roff':1E6}
        self.model('swmod','sw', **self.swmod_params)
        # add a resistive voltage divider to generate the threshold voltage value for switching
        self.R('top', 'vdd', 'trip', 100E6)
        self.R('bot', 'trip', 'vss', 100E6)
        # add swicthes and specify their model parameters
        self.VCS('top', 'bx', 'trip', 'vdd', 'bxl', model='swmod')
        self.VCS('bot', 'trip', 'bx', 'bxl', 'vss', model='swmod')

'''
Ideal DAC SubCircuit Shematic
     refp vdd
      |    |
     --------------------------------------
b0--| |bitlogic0|--bxl0--|              |  |
b1--| |bitlogic1|--bxl1--|  behavioral  |--|--out
... | ...                |voltage source|  |
b5--| |bitlogic5|--bxl5--|              |  |
     --------------------------------------
      |    |
     refn vss
'''
class IdealDac(SubCircuitFactory):
    __name__ = 'IDEALDAC'
    def __init__(self, nbits, **kwargs):
        # number of bits passed as a parameter
        self.nbits = nbits
        # nodes definition based on nbits parameter
        self.__nodes__ = ('refp', 'refn', 'vdd', 'vss') + tuple([('b'+str(i)) for i in range(self.nbits)]) + ('out',)
        super().__init__()
        #ряд битлоджик элеметов
        for i in range(0, self.nbits):
            bitstr = str(i)
            self.X('BL'+bitstr, 'BITLOGIC', 'b'+bitstr, 'b'+bitstr+'l', 'vdd', 'vss')
        #output
        self.voltage_expression = ''.join(['(v(refp)-v(refn))/'+str(2**self.nbits)+'*(']+[('v(b'+str(i)+'l)*'+str(2**i)+'+') for i in range(self.nbits)]+['0)+v(refn)'])
        self.BehavioralSource('out', 'out', 'vss', voltage_expression=self.voltage_expression)

if __name__ == "__main__":
    dac_nbits=4 # specify a number of bits for the DAC
    # operating point analysis
    circuit_op = Circuit('DAC_TestBench') # add an instance of class Circuit to be your testbench
    circuit_op.subcircuit(BitLogic()) # add an instance of BitLogic Subcircuit
    circuit_op.subcircuit(IdealDac(nbits=dac_nbits)) # add an instance of DAC Subcircuit
    circuit_op.X('DAC1', 'IDEALDAC', 'vrefp', 'vrefn', 'vdd', 'vss', ','.join([('b'+str(i)) for i in range(dac_nbits)]), 'out') # connect the DAC subcircuit into the testbench
    circuit_op.V('vdd', 'vdd', circuit_op.gnd, 1) # connect VDD voltage source to vdd node of the DAC subcircuit
    circuit_op.V('vss', 'vss', circuit_op.gnd, 0) # connect VSS voltage source to the ground node of the DAC subcircuit
    circuit_op.V('refp', 'vrefp', circuit_op.gnd, 1) # connect positive reference voltage source to the vrefp node of the DAC subcircuit
    circuit_op.V('refn', 'vrefn', circuit_op.gnd, 0) # connect negative reference voltage source to the vrefn node of the DAC subcircuit
    for i in range(dac_nbits):
        istr=str(i)
        circuit_op.V('b'+istr, 'b'+istr, circuit_op.gnd, 1) # connect nbit number of voltage sources to provide the input code (all '1's in this case)
    simulator_op = circuit_op.simulator(temperature=25, nominal_temperature=25) # add simulator conditions (e.g. temperature)
    analysis_op = simulator_op.operating_point() # specify the analysis to be an operating point analysis
    for node in analysis_op.nodes.values():
        print('Node {}:{:5.2f} V'.format(str(node),float(node))) # print voltage values of all the nodes of the tesbench

    # DAC transfer function - outut voltage vs input code test bench based on operating point analysis
    input_codes = np.arange(2**dac_nbits)
    bin_input_codes = np.zeros(2**dac_nbits)
    outputs = np.zeros(2**dac_nbits)
    for number in input_codes:
        circuit_tf = Circuit('DAC Transfer Function')
        circuit_tf.subcircuit(BitLogic())
        circuit_tf.subcircuit(IdealDac(nbits=dac_nbits))
        circuit_tf.X('DAC1', 'IDEALDAC', 'vrefp', 'vrefn', 'vdd', 'vss', ', '.join([('b'+str(i)) for i in range(dac_nbits)]), 'out')
        print(', '.join([('b'+str(i)) for i in range(dac_nbits)]))
        circuit_tf.V('vdd', 'vdd', circuit_tf.gnd, 1)
        circuit_tf.V('vss', 'vss', circuit_tf.gnd, 0)
        circuit_tf.V('refp', 'vrefp', circuit_tf.gnd, 1)
        circuit_tf.V('refn', 'vrefn', circuit_tf.gnd, 0)
        num_bin_array = [int(num) for num in list(format(number, '0'+str(dac_nbits)+'b'))]
        rev_num_bin_array = list(reversed(num_bin_array))
        for i in range(dac_nbits):
            istr=str(i)
            circuit_tf.V('b'+istr, 'b'+istr, circuit_tf.gnd, 1*rev_num_bin_array[i])
        simulator_tf = circuit_tf.simulator(temperature=25, nominal_temperature=25)
        analysis_tf = simulator_tf.operating_point()
        outputs[number] = float(analysis_tf.nodes['out'])
        print('number: {} - output: {:5.3f}'.format(str(num_bin_array), outputs[number]))
    plt.title(circuit_tf.title)
    plt.xlabel('Input Code')
    plt.ylabel('DAC Output Voltage [V]')
    bin_input_codes = [format(code, '0'+str(dac_nbits)+'b') for code in input_codes]
    plt.xticks(input_codes, bin_input_codes, rotation='vertical')
    plt.plot(input_codes, outputs, 'r--')
    plt.plot(input_codes, outputs, 'ko')
    plt.grid()
    plt.show()
