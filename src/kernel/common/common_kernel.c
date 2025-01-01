#include "common_kernel.h"

void wrap_mram_read(const __mram_ptr void *from, void *to, unsigned int nb_of_bytes) {
  fail_if(nb_of_bytes % 8 != 0, "%d is not a multiple of 8", nb_of_bytes);
  fail_if(nb_of_bytes < 8, "%d is less than 8", nb_of_bytes);
  fail_if(nb_of_bytes > 2048, "%d is greather than 2048", nb_of_bytes);
  fail_if((size_t)from % 8 != 0, "%p FROM is not aligned to 8", from);
  fail_if((size_t)to % 8 != 0, "%p TO is not aligned to 8", to);

  mram_read(from, to, nb_of_bytes);
}