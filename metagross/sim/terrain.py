"""Ground-truth terrain: heightmap + material raster with fast vectorised sampling.

Heights are metres (world z, up). Materials are ``metagross.contracts.scenario.MATERIALS`` ids.
The raster layout follows the scenario ``terrain`` block: rows along +y, columns along +x, the
centre of cell (0, 0) at ``origin_xy``. Queries outside the raster are clamped to the edge cell.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from metagross.sim.geometry import GridSpec


def encode_array_b64(a: np.ndarray, dtype: str) -> str:
    """Little-endian, row-major base64 encoding used by the scenario schema."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode("ascii")


def decode_array_b64(s: str, dtype: str, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`encode_array_b64`; returns a native-endian, writeable array."""
    raw = np.frombuffer(base64.b64decode(s), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(shape).astype(np.dtype(dtype), copy=True)


@dataclass
class Terrain:
    """Heightmap (float32, m) and material ids (uint8) on a :class:`GridSpec`."""

    grid: GridSpec
    height: np.ndarray  # (ny, nx) float32 metres
    material: np.ndarray  # (ny, nx) uint8

    def __post_init__(self) -> None:
        self.height = np.ascontiguousarray(self.height, dtype=np.float32)
        self.material = np.ascontiguousarray(self.material, dtype=np.uint8)
        self._hflat = self.height.ravel()
        self._mflat = self.material.ravel()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_scenario(cls, scenario: dict) -> "Terrain":
        tb = scenario["terrain"]
        ny, nx = (int(v) for v in tb["shape"])
        grid = GridSpec(float(tb["origin_xy"][0]), float(tb["origin_xy"][1]), float(tb["res_m"]), ny, nx)
        h = decode_array_b64(tb["height_b64"], "float32", (ny, nx))
        m = decode_array_b64(tb["material_b64"], "uint8", (ny, nx))
        return cls(grid, h, m)

    # ------------------------------------------------------------------ queries
    def height_at(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Bilinearly interpolated terrain height (m) at world points; any broadcastable shape."""
        g = self.grid
        fi = (np.asarray(y, dtype=np.float64) - g.origin_y) / g.res
        fj = (np.asarray(x, dtype=np.float64) - g.origin_x) / g.res
        fi = np.minimum(np.maximum(fi, 0.0), g.ny - 1.000001)
        fj = np.minimum(np.maximum(fj, 0.0), g.nx - 1.000001)
        i0 = fi.astype(np.int64)
        j0 = fj.astype(np.int64)
        wi = fi - i0
        wj = fj - j0
        base = i0 * g.nx + j0
        h = self._hflat
        h00 = h[base]
        h01 = h[base + 1]
        h10 = h[base + g.nx]
        h11 = h[base + g.nx + 1]
        return (h00 * (1.0 - wj) + h01 * wj) * (1.0 - wi) + (h10 * (1.0 - wj) + h11 * wj) * wi

    def sample_fast(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """float32 fast path for the depth sensor: (bilinear height, nearest material, inside-raster mask).

        Shares one index computation between height and material; points outside the raster are
        clamped (``inside`` tells the caller which ones to discard).
        """
        g = self.grid
        inv = np.float32(1.0 / g.res)
        fi = np.ascontiguousarray((y - np.float32(g.origin_y)) * inv, dtype=np.float32)
        fj = np.ascontiguousarray((x - np.float32(g.origin_x)) * inv, dtype=np.float32)
        inside = (fi >= 0) & (fi <= g.ny - 1) & (fj >= 0) & (fj <= g.nx - 1)
        # cv2.remap: bilinear weights are quantised to 1/32 cell (1.6 mm at 0.05 m) -- negligible.
        h = cv2.remap(self.height, fj, fi, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.remap(self.material, fj, fi, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return h, m, inside

    def material_at(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Material id of the nearest cell (uint8)."""
        i, j = self.grid.xy_to_ij(x, y)
        return self._mflat[i * self.grid.nx + j]

    def gradient(self, smooth_sigma_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Height gradient (dz/dx, dz/dy) on the raster, optionally after Gaussian smoothing."""
        h = self.height
        if smooth_sigma_m > 0.0:
            from scipy.ndimage import gaussian_filter

            h = gaussian_filter(h, smooth_sigma_m / self.grid.res, mode="nearest")
        gy, gx = np.gradient(h, self.grid.res)
        return gx, gy

    def local_median_height(self, x: np.ndarray, y: np.ndarray, half_window_m: float = 0.5, n: int = 9) -> np.ndarray:
        """Median terrain height over an ``n`` x ``n`` sample lattice spanning +-half_window_m around each point.

        Used by the referee's depression test ("more than DEPRESSION_LETHAL_M below the local 1 m
        median"). Sampling a lattice instead of every cell keeps it O(n^2) per query point.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        off = np.linspace(-half_window_m, half_window_m, n)
        ox, oy = np.meshgrid(off, off)
        hs = self.height_at(x[:, None] + ox.ravel()[None, :], y[:, None] + oy.ravel()[None, :])
        return np.median(hs, axis=1)
