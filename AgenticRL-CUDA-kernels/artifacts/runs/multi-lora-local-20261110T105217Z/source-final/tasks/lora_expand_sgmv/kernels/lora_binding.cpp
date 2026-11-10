#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include "../binding_registry.h"
std::vector<torch::Tensor> lora_forward(std::vector<torch::Tensor>,int,bool);
static void register_lora(pybind11::module& m){m.def("lora_forward",&lora_forward);}
REGISTER_BINDING(lora_stage6,register_lora);
