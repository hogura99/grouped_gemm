#include <torch/extension.h>
#include <memory>
#include <map>

namespace grouped_gemm {

struct RawGemmArguments;

#define RawGemmArgumentsPtr std::shared_ptr<RawGemmArguments>

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b);

std::pair<int64_t, RawGemmArgumentsPtr> GetCutlassArguments(
	int num_experts, const torch::Device& device,
	bool trans_a, bool trans_b);

void GroupedGemmWithArgumnts(torch::Tensor a,
	torch::Tensor b,
	torch::Tensor c,
	torch::Tensor batch_sizes,
	bool trans_a, bool trans_b,
	torch::Tensor workspace,
	RawGemmArgumentsPtr arguments);

}  // namespace grouped_gemm
