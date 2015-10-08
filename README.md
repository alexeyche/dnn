# DNN lib and tools
Library of Dynamic Neural Networks for time series related tasks
Main library consist of multithreaded simulator of recurrent spiking (impulse) neural networks dynamics written in C++. It contains variety components which can be connected with each other in different combinations
* Neurons:
    * LeakyIntegrateAndFire - [Adaptive Exponential Integrate-and-fire](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
    * SRMNeuron - [Spike-Response Model](http://www.scholarpedia.org/article/Spike-response_model)
* Synapses:
    * SimpleSynapse - static synapse, can be described by simple exponential decay 
    * DynamicSynapse - synapse with short-term memory ([Tsodyks et al](https://scholar.google.ru/scholar?hl=ru&q=tsodyks+markram+1997&btnG=))
* Activation functions:
    * Determ - Determinate threshold, neuron is firing if membrane reached threshold
    * ExpThreshold - Exponential version of activation function, it has specific increase near threshold value ([Hennequin et al](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3001990/))
* Learning rules:
    * OptimalStdp - Supposed to maximize pre-post information ([Toyoizumi et al](https://scholar.google.ru/citations?view_op=view_citation&hl=ru&user=wUcLR0QAAAAJ&citation_for_view=wUcLR0QAAAAJ:9yKSN-GCB0IC))
    * Stdp - Simple [spike-timing dependent plasticity](http://www.scholarpedia.org/article/STDP)
    * TripleStdp (under development) - Complicated version of Stdp with long range dynamics ([Pfister et al](https://scholar.google.ru/citations?view_op=view_citation&hl=ru&user=mzUYoLgAAAAJ&citation_for_view=mzUYoLgAAAAJ:u5HHmVD_uO8C))
    * MaxLikelihood (under development) - Only works with SRM neuron. It is just maximizing of likelikelhood of spikes, it is makes no sense in unsupervised way, so it needs to be supported by some reward mechanism

# Installation
To install dnn on UNIX like machine you need to satisfy dependencies:
* clang >=3.5 or gcc >=4.9.2
* cmake >=2.8
* libprotobuf-dev
* r-base (for plots and analytics)

To use R API and R scripts you need to install these r packages:
* Rcpp
* zoo
* lattice
* kernlab
* rjson

After satisfying of all prerequisites main installation must be done through installation script:
``` bash
$ git clone https://github.com/alexeyche/dnn
$ cd dnn
$ ./install.sh
```
Script will ask where to install dnn, by default it will use ~/dnn, but you can change it.
Script will build dnn and R package (Rdnn), then trying set through **~/.profile** file environment variables:
``` bash 
export DNN_HOME=~/dnn # or wherever you'd chosen installation path
export LD_LIBRARY_PATH=$DNN_HOME/lib:$LD_LIBRARY_PATH
```

# Getting started

Lets generate some testing time series:
``` R
require(Rdnn)
proto.write.ts(ts.path("test_ts.pb"), list(values=rnorm(1000)))
```
last command will create in **$DNN_HOME/ts** file named **test_ts.pb**

To run simulation you need to set up constants for neccessary neuron specification, e.g. **LeakyIntegrateAndFire** in **$DNN_HOME/user_const.json** file. 
After that you can set up simulation configuration in the same file for that neuron. You need to add spec into **sim_configuration->layers**":
``` json
{
    "size" : 1000,
    "neuron" : "LeakyIntegrateAndFire",
    "act_function" : "Determ",
    "input" : "InputTimeSeries"
}
```
**size** stands for size of the layer, all other entities can be found in specification above: 
* **neuron** for neuron of that layer
* **act_function** function of activation of neuron, for **Determ** threshold can be set in **Determ** key above
* **input** is about providing membrane of neuron with some current specified in input time series. Entity with the name **InputTimeSeries** responsible for this.

In the field **sim_configuration->files** you may find information related to **InputTimeSeries** with macro *@macro-name*. Generaly speakning it awaits path to filename, but this macro can be replaced by file while running simulation with option **--macro-name**

Simulation can be run through python script:
```bash
cd $DNN_HOME
./scripts/run_sim.py --ts-input ./ts/test_ts.pb
```
Option **--ts-input** related to macro in **user_const.json** which allows to point this filename directly into **InputTimeSeries** object.
This run will create in **$DNN_HOME/runs/sim** directory where you can find artefacts of running and png pictures of some introspection of simulation (raster plot)




