SimConfiguration {
	Port: 9090
	Seed: -1
	Jobs: 8
}
Layer {
	IntegrateAndFire {
		TauMem: {{0|1:20}}
		TauRef: {{1|1:20}}
	}
	Sigmoid {
		Threshold: {{2|0.10:0.3}}
		Slope: {{3|100:200}}
	}
    BasicSynapse {
        PspDecay: {{4|1:100}}
    }
    GaussReceptiveField {
    	Sigma: {{5|0.01:2}}}
    	Gain: {{6|0.1:2.0}}
    	LowLevel: -1.5
    	HighLevel: 1.5
    }
}
Layer {
	IntegrateAndFire {
		TauMem: {{7|1:20}}
		TauRef: {{8|1:20}}
	}
	Sigmoid {
		Threshold: {{9|0.10:0.3}}
		Slope: {{10|100:200}}
	}
	Stdp {
		LearningRate: 0.001
	}
}
Connection {
	From: 0 To: 1
	Stochastic {
		Prob: {{11|0.01:1.0}}
	}
	Weight: {{12|0.001:2.0}}
}

Connection {
	From: 0 To: 1
	Stochastic {
		Prob: {{13|0.0:1.0}}
	}
	Weight: {{14|-0.001:-2.0}}
}