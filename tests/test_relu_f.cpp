#include "common.hpp"
#include "test_helper.hpp"

std::vector<float> create_sample_data(size_t size) {
    std::vector<float> buffer(size, 0);
    for (size_t i = 0; i < size; i++) {
        buffer[i] = i-(size/2);
    }
    return buffer;
}


void host_relu(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}


int main(int argc, char **argv) {
    // for(int x = 10; x < 20; x+=10) {
        auto sample_data = create_sample_data(20);
        float *result = new float[sample_data.size()];
        relu_f(sample_data.data(), result, sample_data.size());

        float *result_host = new float[sample_data.size()];
        host_relu(sample_data.data(), result_host, sample_data.size());

        for (size_t i = 0; i < sample_data.size(); i++) {
            if ( abs(result[i]-result_host[i]) > 0.0001 ) {
                RET_TEST_FAIL;
            }
        }
    // }
    RET_TEST_OK;
}
