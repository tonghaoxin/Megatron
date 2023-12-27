#include <torch/extension.h>
#include "authority.h"

namespace parallel_attention {

    bool update() {
        // Note: disable check_authority for now
        /* check_authority(); */
        return false;
    }

} // end namespace parallel_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update", 
            &parallel_attention::update, 
            "parallel_attention update func.");
}
