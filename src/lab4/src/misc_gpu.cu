#include "misc.h"

#include <cuda_runtime.h>
#include <math.h>

__constant__ double d_fact[20] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 
    362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0, 
    87178291200.0, 1307674368000.0, 20922789888000.0, 
    355687428096000.0, 6402373705728000.0, 121645100408832000.0
};

__device__ double misc::wigner_d_device(int l, int m, int s, double costheta) {
    int C1 = max(0, m - s);
    int C2 = min(l + m, l - s);
    double vv = 0.0;
    
    double sinht = sqrt((1.0 - costheta) / 2.0);
    double cosht = sqrt((1.0 + costheta) / 2.0);
    
    for (int t = C1; t <= C2; t++) {
        double sign = ((t - C1) % 2 == 0) ? 1.0 : -1.0;
        
        double denom = d_fact[l + m - t] * d_fact[l - s - t] * d_fact[t] * d_fact[t + s - m];
        
        vv += sign * pow(cosht, 2 * l + m - s - 2 * t) * pow(sinht, 2 * t + s - m) / denom;
    }
    
    return vv * sqrt(d_fact[l + m] * d_fact[l - m] * d_fact[l + s] * d_fact[l - s]);
}