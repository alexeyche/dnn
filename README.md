# dnn
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
