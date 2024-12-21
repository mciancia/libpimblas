#include "libpimblas.h"
#include <gtest/gtest.h>


int add(int a, int b) {
    return a + b;
}

// Test
TEST(AddTest, PositiveNumbers) {
    EXPECT_EQ(add(1, 2), 3);
}

TEST(AddTest, NegativeNumbers) {
    EXPECT_EQ(add(-1, -2), -3);
}

TEST(AddTest, Zero) {
    EXPECT_EQ(add(0, 0), 0);
}