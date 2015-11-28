#pragma once

#include <dnn/connections/connection.h>
#include <dnn/protos/difference_of_gaussians.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct DifferenceOfGaussiansC : public Serializable<Protos::DifferenceOfGaussiansC> {
    DifferenceOfGaussiansC() : sigma_pos(1.0), sigma_neg(3.0), neg_amp(1.0), dimension(2), apply_amplitude(true) {}

    void serial_process() {
        begin() << "sigma_pos: " << sigma_pos << ", "
                << "sigma_neg: " << sigma_neg << ", "
                << "neg_amp: " << neg_amp << ", "
        		<< "dimension: " << dimension << ", "
                << "apply_amplitude: " << apply_amplitude << Self::end;
    }

    int dimension;
    double sigma_pos;
    double sigma_neg;
    double neg_amp;
    bool apply_amplitude;
};


class DifferenceOfGaussians : public Connection<DifferenceOfGaussiansC> {
public:
    static double gaussFunction2d(double x, double y, double xc, double yc, double sigma) {
    	return (1/(sqrt(2*PI)*sigma)) * exp( - ( (x-xc)*(x-xc) + (y-yc)*(y-yc) )/(2*sigma*sigma) ) ;
    }
    static double gaussFunction(double x, double xc, double sigma) {
        return (1/(sqrt(2*PI)*sigma)) * exp( - ( (x-xc)*(x-xc) )/(2*sigma*sigma) ) ;
    }
    ConnectionRecipe getConnectionRecipe(const SpikeNeuronBase &left, const SpikeNeuronBase &right) {
    	ConnectionRecipe recipe;
    	double v = 0.0;
        if(c.dimension == 1) {
            int right_circled = right.localId()-getPostLayerSize();
            int left_circled = left.localId()-getPreLayerSize();
            v += gaussFunction(left.localId(), right.localId(), c.sigma_pos) -
    				  c.neg_amp * gaussFunction(left.localId(), right.localId(), c.sigma_neg);
            v += gaussFunction(left.localId(), right_circled, c.sigma_pos) -
                      c.neg_amp * gaussFunction(left.localId(), right_circled, c.sigma_neg);
            v += gaussFunction(left_circled, right.localId(), c.sigma_pos) -
                      c.neg_amp * gaussFunction(left_circled, right.localId(), c.sigma_neg);

        } else
        if(c.dimension == 2) {
            int right_xi_circled = right.xi()-right.colSize();
            int left_xi_circled = left.xi()-left.colSize();
            int right_yi_circled = right.yi()-right.colSize();
            int left_yi_circled = left.yi()-left.colSize();

            v += gaussFunction2d(right.xi(), right.yi(), left.xi(), left.yi(), c.sigma_pos) -
                      c.neg_amp * gaussFunction2d(right.xi(), right.yi(), left.xi(), left.yi(), c.sigma_neg);
            v += gaussFunction2d(right.xi(), right.yi(), left_xi_circled, left_yi_circled, c.sigma_pos) -
                      c.neg_amp * gaussFunction2d(right.xi(), right.yi(), left_xi_circled, left_yi_circled, c.sigma_neg);
            v += gaussFunction2d(right_xi_circled, right_yi_circled, left.xi(), left.yi(), c.sigma_pos) -
                      c.neg_amp * gaussFunction2d(right_xi_circled, right_yi_circled, left.xi(), left.yi(), c.sigma_neg);
        } else {
            throw dnnException() << "Can't build DifferenceOfGaussians with dimension like this: " << c.dimension << "\n"
                                 << "Only 1 or 2 dimension supported\n";
        }
    	recipe.exists = true;
    	if(v<0) {
    		recipe.inhibitory = true;
    	}
        if(c.apply_amplitude) {
            recipe.amplitude = fabs(v);
        }
    	return recipe;
    }
};





};