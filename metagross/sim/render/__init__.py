"""Three.js stereo renderer for the METAGROSS simulator (world side only).

``ThreeRenderer`` (bridge.py) implements ``metagross.contracts.interfaces.RendererProto``
by driving Chrome/Edge + WebGL2 through Playwright; ``camera_model`` is the Python
pinhole/pose reference the renderer is tested against; ``testscene`` builds small
synthetic scenarios for tests and benchmarks. The autonomy stack must never import
this package.
"""

from metagross.sim.render.bridge import RendererUnavailable, ThreeRenderer

__all__ = ["ThreeRenderer", "RendererUnavailable"]
