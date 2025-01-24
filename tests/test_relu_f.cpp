#include "common.hpp"
#include "test_helper.hpp"

#include <iomanip>
#include <limits>

std::vector<float> create_sample_data(size_t size) {
    std::vector<float> buffer(size, 0);
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (float)i-((float)size/2);
    }
    return buffer;
}


void host_relu(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}


int main(int argc, char **argv) {
    auto sample_data = create_sample_data(64000);
    float *result = new float[sample_data.size()];
    relu_f(sample_data.data(), result, sample_data.size());

    float *result_host = new float[sample_data.size()];
    host_relu(sample_data.data(), result_host, sample_data.size());
    for (size_t i = 0; i < sample_data.size(); i++) {
        if ( abs(result[i]-result_host[i]) > 0.01 ) {
            for(int j = 0; j < sample_data.size(); j++){
                std::cout << 
                    std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) <<
                    "Host: " << result_host[j] << 
                    " DPU: " << result[j] << 
                    " Diff: " << abs(result[j]-result_host[j]) <<  std::endl;
            }
            RET_TEST_FAIL;
        }
    }
    RET_TEST_OK;
}
