#pragma once

#include "vinyaikina_e_multidimensional_integrals_simpson_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace vinyaikina_e_multidimensional_integrals_simpson_method {

  class VinyaikinaEMultidimIntegrSimpsonSEQ : public BaseTask {
  public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kSEQ;
    }
    explicit VinyaikinaEMultidimIntegrSimpsonSEQ(const InType& in);

  private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;

    double I_res;
  };

}  // namespace vinyaikina_e_multidimensional_integrals_simpson_method
