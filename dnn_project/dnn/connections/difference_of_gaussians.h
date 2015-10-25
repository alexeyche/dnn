#pragma once

#include <dnn/connections/connection.h>
#include <dnn/protos/difference_of_gaussians.pb.h>

namespace dnn {

/*@GENERATE_PROTO@*/
struct DifferenceOfGaussiansC : public Serializable<Protos::DifferenceOfGaussiansC> {
    DifferenceOfGaussiansC() : a(3.0), b(3.0), r(1), dimension(2), apply_amplitude(true) {}

    void serial_process() {
        begin() << "a: " << a << ", "
                << "b: " << b << ", "
        		<< "r: " << r << ", "
                << "dimension: " << dimension << ", "
                << "apply_amplitude: " << apply_amplitude << Self::end;
    }

    int dimension;
    double a;
    double b;
    double r;
    bool apply_amplitude;
};


class DifferenceOfGaussians : public Connection<DifferenceOfGaussiansC> {
public:
    const string name() const {
        return "DifferenceOfGaussians";
    }
    static double gaussFunction2d(double x, double y, double xc, double yc, double sigma) {
    	return exp( - ( (x-xc)*(x-xc) + (y-yc)*(y-yc) )/(2*sigma*sigma) ) ;
    }
    static double gaussFunction(double x, double xc, double sigma) {
        return exp( - ( (x-xc)*(x-xc) )/(2*sigma*sigma) ) ;
    }
    ConnectionRecipe getConnectionRecipe(const SpikeNeuronBase &left, const SpikeNeuronBase &right) {
    	ConnectionRecipe recipe;
    	double v = 0.0;
        if(c.dimension == 1) {
            int right_circled = right.localId()-getPostLayerSize();
            int left_circled = left.localId()-getPreLayerSize();
            v += (1.0+c.a) * gaussFunction(left.localId(), right.localId(), c.r) - 
    				  c.a * gaussFunction(left.localId(), right.localId(), c.b*c.r);
            v += (1.0+c.a) * gaussFunction(left.localId(), right_circled, c.r) - 
                      c.a * gaussFunction(left.localId(), right_circled, c.b*c.r);
            v += (1.0+c.a) * gaussFunction(left_circled, right.localId(), c.r) - 
                      c.a * gaussFunction(left_circled, right.localId(), c.b*c.r);
            
        } else
        if(c.dimension == 2) {
            int right_xi_circled = right.xi()-right.colSize();
            int left_xi_circled = left.xi()-left.colSize();
            int right_yi_circled = right.yi()-right.colSize();
            int left_yi_circled = left.yi()-left.colSize();
            
            v += (1.0+c.a) * gaussFunction2d(right.xi(), right.yi(), left.xi(), left.yi(), c.r) - 
                      c.a * gaussFunction2d(right.xi(), right.yi(), left.xi(), left.yi(), c.b*c.r);
            v += (1.0+c.a) * gaussFunction2d(right.xi(), right.yi(), left_xi_circled, left_yi_circled, c.r) - 
                      c.a * gaussFunction2d(right.xi(), right.yi(), left_xi_circled, left_yi_circled, c.b*c.r);
            v += (1.0+c.a) * gaussFunction2d(right_xi_circled, right_yi_circled, left.xi(), left.yi(), c.r) - 
                      c.a * gaussFunction2d(right_xi_circled, right_yi_circled, left.xi(), left.yi(), c.b*c.r);
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