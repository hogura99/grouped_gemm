#include "grouped_gemm.h"

#include <torch/torch.h>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("get_cutlass_arguments", &GetCutlassArguments, "Get Cutlass arguments.");
  m.def("gmm_with_arguments", &GroupedGemmWithArgumnts, "Grouped GEMM with arguments.");
  
  // py::class_<RawGemmArguments>(m, "RawGemmArguments");
}

}  // namespace grouped_gemm
