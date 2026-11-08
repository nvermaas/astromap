import sys
import numpy as np
import OpenEXR
import Imath
from PIL import Image

def exr_to_jpg(exr_path: str, jpg_path: str,
              gamma: float = 2.2, percentile: float = 99.9) -> None:
    with OpenEXR.File(exr_path) as exr:
        # Channels are typically "R", "G", "B"; some maps use "A" or others
        chans = exr.channels()
        names = [c for c in ("R", "G", "B") if c in chans] or list(chans)[:3]

        # Read as float32 and stack into (H, W, 3)
        arrays = []
        header = exr.header()
        for name in names:
            ch = exr.channels(name, Imath.PixelType(Imath.PixelType.FLOAT))
            arrays.append(np.frombuffer(ch, dtype=np.float32))
        data = np.stack(arrays, axis=-1).reshape(header["dataWindow"].max.y + 1,
                                                header["dataWindow"].max.x + 1, 3)

    # Data is linear HDR: apply an asinh stretch, clip at a high percentile
    # to keep bright stars from washing out the faint Milky Way, then gamma.
    lo, hi = np.percentile(data, 50), np.percentile(data, percentile)
    scaled = np.arcsinh(np.clip((data - lo) / max(hi - lo, 1e-9), 0, 1))
    scaled = scaled / np.nanmax(scaled)          # normalise to [0, 1]
    rgb = np.clip(scaled, 0, 1) ** (1 / gamma)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(jpg_path)

if __name__ == "__main__":
    exr_to_jpg(sys.argv[1], sys.argv[2])