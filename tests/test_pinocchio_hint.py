import platform
import sys

import pytest


@pytest.mark.skipif(
    platform.system() == "Linux",
    reason="On Linux SAPIEN falls back to its built-in pinocchio, so PinocchioModel is never None",
)
def test_pinocchio_missing_raises_helpful_importerror(monkeypatch):
    """When the optional pinocchio package is missing (e.g. on MacOS),
    importing the IK kinematics utilities should raise an ImportError that
    tells the user how to install pinocchio (see issue #1267)."""
    # simulate pinocchio being absent: None in sys.modules makes `import pinocchio` fail
    monkeypatch.setitem(sys.modules, "pinocchio", None)
    # drop cached modules so the fallback logic in SAPIEN's wrapper re-runs
    monkeypatch.delitem(sys.modules, "sapien.wrapper.pinocchio_model", raising=False)
    monkeypatch.delitem(
        sys.modules, "mani_skill.agents.controllers.utils.kinematics", raising=False
    )
    with pytest.raises(ImportError, match="pinocchio is not installed"):
        import mani_skill.agents.controllers.utils.kinematics  # noqa: F401
