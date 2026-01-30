#include "utils.cuh"

int main(int argc, char const **args) {


  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif
  return 0;
}
