#include "common.h"
#include "share_vector_add.h"

#define INTROSPECT 1

void distribute_data_to(struct dpu_set_t *dpu_set, const char *symbol, void *array_start, size_t array_item_size,
                        uint32_t offset) {
  unsigned int i = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(*dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, array_start + (array_item_size)*i)); }
  DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, symbol, offset, array_item_size, DPU_XFER_DEFAULT));
}

int NotAlignedTo8(void *p) { return (size_t)p % 8 != 0; }

int vector_add(const int *a_input_ptr, const int *b_input_ptr, size_t num_elem, int *output) {
  show_info("vector_add size=%lu", num_elem);

  show_trace("a_input_ptr=[%p]  b_input_ptr=[%p]  output=[%p]", a_input_ptr, b_input_ptr, output);

  if (NotAlignedTo8(a_input_ptr) || NotAlignedTo8(b_input_ptr) || NotAlignedTo8(output)) {
    show_error("input/output is not aligned to 8 ");
  }

  struct dpu_set_t dpu_set, dpu;

  uint32_t dpus_exe = 8;
  DPU_ASSERT(dpu_alloc(dpus_exe, NULL, &dpu_set));
  char *kern_name = pimblas_get_kernel_dir_concat_free("vector_add.kernel");
  show_debug("kern_path=%s\n", kern_name);

  DPU_ASSERT(dpu_load(dpu_set, kern_name, NULL));
  free(kern_name);

  uint32_t nr_of_dpus;
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
  show_debug("Allocated %d DPU(s)\n", nr_of_dpus);

  dpu_arguments_t input_arguments[nr_of_dpus];

  // check fail modulo  2621440;
  //  16     x  4
  const unsigned int input_size_8bytes = ((num_elem * sizeof(T)) % 8) != 0 ? roundup(num_elem, 8) : num_elem;  // 16

  // const unsigned int input_size_dpu = divceil(num_elem, nr_of_dpus); // div
  // up

  // const unsigned int input_size_dpu_8bytes = ((input_size_dpu * sizeof(T)) %
  // 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu;

  const unsigned int num_items = num_elem / nr_of_dpus;
  const unsigned int transfer = (input_size_8bytes / (nr_of_dpus * 2)) * 8;

  unsigned int i = 0;
  for (i = 0; i < nr_of_dpus - 1; i++) {
    input_arguments[i].items = num_items;
    input_arguments[i].transfer_block_size8 = transfer;
    input_arguments[i].kernel = 0;
  }
  input_arguments[nr_of_dpus - 1].items = num_items;
  input_arguments[nr_of_dpus - 1].transfer_block_size8 = transfer;
  input_arguments[nr_of_dpus - 1].kernel = 0;

  // for(i=0;i<nr_of_dpus;i++)
  // {
  //   printf("[host]: {arguments[%d].size=[%d].transfer_size=[%d].kern=[%d]}
  //   (sizeof(T)[%d])*(num_elem[%d])=[%d], [%d][%d][%d] \n", i,
  //   input_arguments[i].size,
  //   input_arguments[i].transfer_size,input_arguments[i].kernel,sizeof(T),num_elem,
  //   sizeof(T)*num_elem,input_size_8bytes,input_size_dpu,input_size_dpu_8bytes);
  // }

  // send arguments
  distribute_data_to(&dpu_set, "DPU_INPUT_ARGUMENTS", input_arguments, sizeof(input_arguments[0]), 0);

  distribute_data_to(&dpu_set, DPU_MRAM_HEAP_POINTER_NAME, a_input_ptr, transfer, 0);
  distribute_data_to(&dpu_set, DPU_MRAM_HEAP_POINTER_NAME, b_input_ptr, transfer, transfer);

  //  DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu,
  //  &input_arguments[i])); } DPU_ASSERT(dpu_push_xfer(dpu_set,
  //  DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]),
  //  DPU_XFER_DEFAULT));

  //  DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu,
  //  a_input_ptr + input_arguments[i].num_items * i)); }
  //  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
  //  DPU_MRAM_HEAP_POINTER_NAME, 0, input_arguments[0].transfer_size_bytes,
  //  DPU_XFER_DEFAULT));

  //   DPU_FOREACH(dpu_set, dpu, i) {

  //   DPU_ASSERT(dpu_prepare_xfer(dpu, b_input_ptr +
  //   input_arguments[i].num_items * i));

  //   }

  //   DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
  //   DPU_MRAM_HEAP_POINTER_NAME, input_arguments[0].transfer_size_bytes,
  //   input_arguments[0].transfer_size_bytes, DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

#ifdef INTROSPECT

  unsigned int each_dpu = 0;
  show_debug("Display DPU Logs");
  DPU_FOREACH(dpu_set, dpu) {
    show_debug("DPU {}", each_dpu);
    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    each_dpu++;
  }
#endif

  //  DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, output +
  //  input_arguments[i].num_items * i));  } DPU_ASSERT(dpu_push_xfer(dpu_set,
  //  DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
  //  input_arguments[i].transfer_size_bytes,
  //  input_arguments[i].transfer_size_bytes, DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_free(dpu_set));

  return 0;
}
