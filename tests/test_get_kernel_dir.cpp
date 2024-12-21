#include "libpimblas.h"
#include <gtest/gtest.h>


int add(int a, int b) {
    return a + b;
}


TEST(KernelDefaultDir, NotNull) {
     EXPECT_NE(pimblas_get_kernel_dir(), nullptr);
}




// TEST(AddTest, PositiveNumbers) {
//     EXPECT_EQ(add(1, 2), 3);
// }

// TEST(AddTest, NegativeNumbers) {
//     EXPECT_EQ(add(-1, -2), -3);
// }

// TEST(AddTest, Zero) {
//     EXPECT_EQ(add(0, 0), 0);
// }


// TEST(showhello, first) {
//      pimblas_test_function(5); 
//      gemv(5);
//      EXPECT_EQ(0, 0);
// }

