#include <torch/extension.h>
#include "authority.h"

namespace parallel_attention {

    bool update() {
        check_authority();
        return true;
    }

} // end namespace parallel_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", 
            &parallel_attention::update, 
            "parallel_attention update func.");
}
