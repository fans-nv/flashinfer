"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest

from .test_aot_sm103_modules import _registered_generators

LOCALIZED = "gen_trtllm_gen_fused_moe_localized_module"


CAPS = {
    "sm107": {"sm107": True, "sm107a_exact": True},
    "sm100": {"sm100": True, "sm100a_exact": True, "sm100f": True},
}


# The split-N BMM package is private: the localized module is AOT-listed only on SM107 and only
# where the package is staged.
@pytest.mark.parametrize("staged", [False, True], ids=["unstaged", "staged"])
@pytest.mark.parametrize("arch", ["sm107", "sm100"])
def test_localized_module_is_aot_listed_only_with_the_package(
    monkeypatch, arch, staged
):
    registered = _registered_generators(
        monkeypatch,
        CAPS[arch],
        overrides={"has_trtllm_gen_bmm_split_n": lambda: staged},
    )
    assert "gen_trtllm_gen_fused_moe_sm100_module" in registered
    assert (LOCALIZED in registered) == (arch == "sm107" and staged)
