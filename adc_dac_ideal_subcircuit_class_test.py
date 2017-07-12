#!/usr/bin/env python
"""
adc_dac_ideal_subcircuit_class_test.py: set of classes for an ideal
N-bit ADC/DAC modelling with PySpice
"""
__author__      = "Olzhas S. Tazabekov"

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
        self.swmod_params = {'ron':1E-2, 'roff':1E6}
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
        # add an nbit number of BitLogic subcircuits
        for i in range(0, self.nbits):
            bitstr = str(i)
            self.X('BL'+bitstr, 'BITLOGIC', 'b'+bitstr, 'b'+bitstr+'l', 'vdd', 'vss')
        # make an output voltage expression based on nbits parameter and pass this string into added BehavioralSource
        self.voltage_expression = ''.join(['(v(refp)-v(refn))/'+str(2**self.nbits)+'*(']+[('v(b'+str(i)+'l)*'+str(2**i)+'+') for i in range(self.nbits)]+['0)+v(refn)'])
        self.BehavioralSource('out', 'out', 'vss', voltage_expression=self.voltage_expression)

'''
Ideal Sample and Holde Subcircuit

          vdd
           |
         --------------------------------------------------------------------------
        |     -----------                                           ------------   |
    in--|    |           |                                         |            |  |
        |    |  -------  |                                         |  --------  |  |
   clk--|     -|-      | |                                          -|-       | |  |
        |      | inbuf |----inbuf--\ ---inS-------\ ---outS------|   | outbuf |----|--out
  trip--|  in--|+      |           |        |     |         |    |---|+       |    |
        |       -------            |      |Cap|   |       |Cap|       --------     |
        |                         clk_      |    clk        |                      |
        |                                   vss             vss                    |
         --------------------------------------------------------------------------
           |
          vss
'''
class IdealSampleHold(SubCircuitFactory):
    __name__ = 'IDEALSAMPLEHOLD'
    __nodes__ = ('in', 'trip', 'clk', 'vdd', 'vss', 'out')
    def __init__(self, **kwargs):
        super().__init__()
        #switches model
        self.swmod_params = {'ron':1E-2, 'roff':1E6}
        self.model('swmod','sw', **self.swmod_params)

        #элементы схемотехники Сэмпл Холд
        self.VCVS('in', 'in', 'inbuf', 'inbuf', 'vss', voltage_gain=100E6)
        self.VCS('1', 'trip', 'clk', 'inbuf', 'inS', model='swmod')
        self.C('s1', 'inS', 'vss', 1E-10)
        self.VCS('2', 'clk', 'trip', 'inS', 'outS', model='swmod')
        self.C('outS', 'outS', 'vss', 1E-12)
        self.VCVS('out', 'outS', 'out', 'out', 'vss', voltage_gain=100E6)


'''
Ideal Pipeline stage
1. Сигнал из Sample/Hold сравнивается с референсом в компараторе.
2. если инпут больше референса - на оутпут выводится 1, референс вычитается из инпута и умножается на 2 и передается сл элементу
   если инпут меньше референса - на оутпут вывподится 0, инпут умножается на 2 и передается сл элементу
3. Повторить
'''

class IdealPipelineStage(SubCircuitFactory):
    __name__ = 'IDEALPIPELINESTAGE'
    __nodes__ = ('in', 'cm', 'trip','vdd', 'vss', 'bitout', 'out')
    def __init__(self, **kwargs):
        super().__init__()
        #Swiches models
        self.swmod_params = {'ron':1E-2, 'roff':1E6}
        self.model('swmod','sw', **self.swmod_params)
        #Элементы схемотехники Пайплайн Стейдж
        self.VCS('1', 'in', 'cm', 'vdd', 'bitout', model='swmod')
        self.VCS('2', 'cm', 'in', 'vss', 'bitout', model='swmod')
        self.VCVS('outh', 'in', 'cm', 'vinh', 'vss', voltage_gain=2)
        self.VCVS('outl', 'in', 'vss', 'vinl', 'vss', voltage_gain=2)
        self.VCS('3', 'bitout', 'trip', 'vinh', 'out', model='swmod')
        self.VCS('4', 'trip', 'bitout', 'vinl', 'out', model='swmod')


'''
Ideal ADC Subcircuit

            vdd
             |
           -------------------------------------------------------------------------------------------------------------------------
          |                                                                                                                         |
    refp--|-|                                                                                                                       |--bitout1
          | cm-----                    refp---    refn---                      --bitout1        --bitout2             --bitoutN     |
    refn--|-|      |                          |          |                    |                |                     |              |--bitout2
          |     |Sample/Hold|--outsh--|levelshift by lsb/2 & refn|--pipin--|stage1|--out1--|stage2|--out2-- ..N ..|stageN|--outN    |
     clk--|--------|   |                                                                                                            |-- ...
          |            |                                                                                                            |
      in--|------------                                                                                                             |--bitoutN
          |                                                                                                                         |
           -------------------------------------------------------------------------------------------------------------------------
             |
            vss
'''
class IdealAdc(SubCircuitFactory):
    __name__ = 'IDEALADC'
    def __init__(self, nbits, **kwargs):
        #number of bits
        self.nbits = nbits
        self.__nodes__ = ('refp', 'refn', 'vdd', 'vss', 'in', 'clk') + tuple([('b'+str(i)) for i in range(self.nbits)])
        super().__init__()
        #common voltage definition = (refp+refn)/2
        self.BehavioralSource('cm', 'cm', self.gnd, voltage_expression='(v(refp)+v(refn))/2')
        #logic switching point
        self.R('top', 'vdd', 'trip', 10E6)
        self.R('bot', 'trip', 'vss', 10E6)
        #ideal sample and hold
        self.X('idealsamplehold', 'IDEALSAMPLEHOLD', 'in', 'trip', 'clk', 'vdd', 'vss', 'outsh')
        #levelshift by refn and 1/2LSB
        self.BehavioralSource('pip', 'pipin', 'vss', voltage_expression='v(outsh)-v(refn)+((v(refp)-v(refn))/2^'+str(self.nbits+1)+')')
        #Каскад пайплайн стейдж
        self.X('idealpipelinestage'+str(self.nbits-1), 'IDEALPIPELINESTAGE', 'pipin', 'cm', 'trip', 'vdd', 'vss', 'b'+str(self.nbits-1), 'out'+str(self.nbits-1))
        for i in range(self.nbits-1, 0, -1):
            bitstr = str(i)
            prev_bitstr = str(i-1)
            self.X('idealpipelinestage'+prev_bitstr, 'IDEALPIPELINESTAGE', 'out'+bitstr, 'cm', 'trip', 'vdd', 'vss', 'b'+prev_bitstr, 'out'+prev_bitstr)

if __name__ == "__main__":
    #ADC/DAC Transient Test Bench Diagram
    '''
                     ---        ---
                    |   |--b0--|   |
        |vsin|--in--|ADC|--..--|DAC|--out
               clk--|   |--bn--|   |
                     ---        ---
    '''
    #ADC/DAC Transient Test Bench Vars
    dac_nbits = 6
    adc_nbits = 6
    f_signal = 10E6
    amp_signal = 0.5
    dc_signal = 0.5
    f_sample = 100E6
    t_sample = 1/f_sample
    #Simulator Vars
    tran_sim_end_time = 1E2/f_signal

    #Test Bench Definition
    circuit = Circuit('TestBench')
    circuit.subcircuit(IdealDac(nbits=dac_nbits))
    circuit.subcircuit(BitLogic())
    circuit.subcircuit(IdealSampleHold())
    circuit.subcircuit(IdealPipelineStage())
    circuit.subcircuit(IdealAdc(nbits=adc_nbits))
    #Instantiating Sources
    # -input source
    circuit.Sinusoidal('in1', 'in1', circuit.gnd, dc_offset=dc_signal, offset=dc_signal, amplitude=amp_signal, frequency=f_signal, delay=0, damping_factor=0)
    # -supply rails and references
    circuit.V('vdd', 'vdd', circuit.gnd, 1)
    circuit.V('refp', 'vrefp', circuit.gnd, 1)
    circuit.V('refn', 'vrefn', circuit.gnd, 0)
    # - clock
    circuit.Pulse('clk', 'clk', circuit.gnd, 0, 1, t_sample/2, t_sample, fall_time=0.01*t_sample, rise_time=0.01*t_sample)
    # -ADC/DAC
    circuit.X('ADC', 'IDEALADC', 'vrefp', 'vrefn', 'vdd', circuit.gnd, 'in1', 'clk', ', '.join([('b'+str(i)) for i in range(adc_nbits)]))
    circuit.X('DAC', 'IDEALDAC', 'vrefp', 'vrefn', 'vdd', circuit.gnd, ', '.join([('b'+str(i)) for i in range(dac_nbits)]), 'out')
    #Transient Sims
    # -instantiating a simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    # -defining Simulator Properties
    simulator.options('MAXORD=5')
    simulator.options('METHOD=Gear')
    # -Extractin Analysis Data
    analysis = simulator.transient(step_time=1E-9, end_time=tran_sim_end_time)
    #Printing the Circuit Nodes (optional)
    #print(analysis.nodes.values())

    #Plotting Transient Sims Data

    f, axarr = plt.subplots(adc_nbits+2, sharex=True)
    f.tight_layout()
    axarr[0].set_title('ADC Input and DAC Output')
    axarr[0].plot(analysis.time, np.array(analysis.in1))
    axarr[0].plot(analysis.time, np.array(analysis.out))
    axarr[1].set_title('ADC Clock and Trip Voltage')
    axarr[1].plot(analysis.time, np.array(analysis.clk))
    axarr[1].plot(analysis.time, np.array(analysis.nodes['xadc.trip']))
    for i in range(adc_nbits):
        axarr[i+2].set_title('ADC Bit '+str(i))
        axarr[i+2].plot(analysis.time, np.array(analysis.nodes['b'+str(i)]))
    axarr[len(axarr)-1].set_xlabel('time [s]')

    #Ramp-up Test Bench Definition
    circuit_pwl = Circuit('TestBench')
    circuit_pwl.subcircuit(IdealDac(nbits=dac_nbits))
    circuit_pwl.subcircuit(BitLogic())
    circuit_pwl.subcircuit(IdealSampleHold())
    circuit_pwl.subcircuit(IdealPipelineStage())
    circuit_pwl.subcircuit(IdealAdc(nbits=adc_nbits))
    #Instantiating Sources
    # -input source
    circuit_pwl.V('in1', 'in1', circuit.gnd, 'pwl(0 0 '+str(tran_sim_end_time)+' 1 r=0')
    # -supply rails and references
    circuit_pwl.V('vdd', 'vdd', circuit.gnd, 1)
    circuit_pwl.V('refp', 'vrefp', circuit.gnd, 1)
    circuit_pwl.V('refn', 'vrefn', circuit.gnd, 0)
    # - clock
    circuit_pwl.Pulse('clk', 'clk', circuit.gnd, 0, 1, t_sample/2, t_sample, fall_time=0.01*t_sample, rise_time=0.01*t_sample)
    # -ADC/DAC
    circuit_pwl.X('ADC', 'IDEALADC', 'vrefp', 'vrefn', 'vdd', circuit.gnd, 'in1', 'clk', ', '.join([('b'+str(i)) for i in range(adc_nbits)]))
    circuit_pwl.X('DAC', 'IDEALDAC', 'vrefp', 'vrefn', 'vdd', circuit.gnd, ', '.join([('b'+str(i)) for i in range(dac_nbits)]), 'out')
    #Transient Sims
    # -instantiating a simulator
    simulator_pwl = circuit_pwl.simulator(temperature=25, nominal_temperature=25)
    # -defining Simulator Properties
    simulator_pwl.options('MAXORD=5')
    simulator_pwl.options('METHOD=Gear')
    # -Extractin Analysis Data
    analysis_pwl = simulator_pwl.transient(step_time=1E-9, end_time=tran_sim_end_time)
    #Printing the Circuit Nodes (optional)
    #print(analysis.nodes.values())

    #Plotting Transient Sims Data
    f_pwl, axarr_pwl = plt.subplots(adc_nbits+2, sharex=True)
    f_pwl.tight_layout()
    axarr_pwl[0].set_title('ADC Input and DAC Output')
    axarr_pwl[0].plot(analysis_pwl.time, np.array(analysis_pwl.in1))
    axarr_pwl[0].plot(analysis_pwl.time, np.array(analysis_pwl.out))
    axarr_pwl[1].set_title('ADC Clock and Trip Voltage')
    axarr_pwl[1].plot(analysis_pwl.time, np.array(analysis_pwl.clk))
    axarr_pwl[1].plot(analysis_pwl.time, np.array(analysis_pwl.nodes['xadc.trip']))
    for i in range(adc_nbits):
        axarr_pwl[i+2].set_title('ADC Bit '+str(i))
        axarr_pwl[i+2].plot(analysis_pwl.time, np.array(analysis_pwl.nodes['b'+str(i)]))
    axarr_pwl[len(axarr)-1].set_xlabel('time [s]')

    #Quantization Noise. Fourier Transform of the ADC input and DAC Output
    N = np.array(analysis.in1).size
    T = analysis.time[1]-analysis.time[0]
    print('N: '+str(N))
    print('T: '+str(T))
    signal = np.array(analysis.in1)
    fft_signal = np.abs(np.fft.fft(signal))
    max_fft_signal = np.max(fft_signal)
    output = np.array(analysis.out)
    fft_output = np.abs(np.fft.fft(output))
    max_fft_output = np.max(fft_output)
    freq_fft = np.linspace(0, 1/(2*T), N//2)
    f_q, axarr_q = plt.subplots(2)
    f_q.tight_layout()
    axarr_q[0].set_title('ADC Input and DAC Output Noise Floor. Frequency Domain ')
    axarr_q[0].plot(freq_fft[:N//2], 20*np.log10(2/N*fft_signal[:N//2]))
    axarr_q[0].plot(freq_fft[:N//2], 20*np.log10(2/N*fft_output[:N//2]))
    axarr_q[1].set_title('ADC Input and DAC Output. Time Domain')
    axarr_q[1].plot(analysis.time, signal)
    axarr_q[1].plot(analysis.time, output)
    plt.grid(True)
    plt.show()
