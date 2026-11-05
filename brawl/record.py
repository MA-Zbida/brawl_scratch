"""The one recording format (research/DESIGN.md section 10).

Human demonstrations, agent experience and evaluation all write the same thing: a folder of
shards, one row per recorded step, plus ``meta.json``.

    <recording>/
      meta.json           what was recorded: regime, setting, columns, frame encoding, row count
      shard_00000.npz     the columns of up to ``shard_rows`` rows, and their frames
      stills/<row>.png    full-resolution, lossless frames for labelling (optional)

Columns are named arrays, one value per row; which columns exist depends on the writer (a human
demonstration has ``t``, ``keys`` and ``pressed``; agent steps add ``obs``, ``action``, ``reward``
...). Frames are downscaled and stored as JPEG bytes (half resolution: about 70 KB a frame, 15 GB an
hour at 60 Hz, 2.8 ms to encode);
anything derived from them (crops, features, labels) can be recomputed later without new data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

FORMAT = 1


def encode(image: np.ndarray, scale: float, quality: int) -> bytes:
    import cv2

    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


def decode(data: bytes | np.ndarray) -> np.ndarray:
    import cv2

    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


class Writer:
    """Appends rows and writes a shard every ``shard_rows`` rows. Use as a context manager."""

    def __init__(self, root: Path, meta: dict[str, Any], *, shard_rows: int = 600,
                 scale: float = 0.5, quality: int = 90) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)  # never write into an existing recording
        self.meta = dict(meta)
        self.shard_rows, self.scale, self.quality = shard_rows, scale, quality
        self.rows = 0
        self.shards = 0
        self.stills = 0
        self._columns: dict[str, list[Any]] = {}
        self._frames: list[bytes] = []

    def append(self, frame: Optional[np.ndarray], **columns: Any) -> int:
        """Add one row; returns its index. Every row must have the same columns."""
        if self._columns and set(columns) != set(self._columns):
            raise ValueError(f"row columns {sorted(columns)} differ from {sorted(self._columns)}")
        for name, value in columns.items():
            self._columns.setdefault(name, []).append(value)
        self._frames.append(b"" if frame is None else encode(frame, self.scale, self.quality))
        self.rows += 1
        if len(self._frames) >= self.shard_rows:
            self.flush()
        return self.rows - 1

    def still(self, row: int, image: np.ndarray) -> None:
        """Save a full-resolution lossless copy of the frame of ``row``."""
        import cv2

        folder = self.root / "stills"
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f"{row:07d}.png"), image)
        self.stills += 1

    def flush(self) -> None:
        if not self._frames:
            return
        sizes = np.array([len(f) for f in self._frames], dtype=np.int64)
        arrays = {name: np.asarray(values) for name, values in self._columns.items()}
        np.savez(self.root / f"shard_{self.shards:05d}.npz",
                 frame_bytes=np.frombuffer(b"".join(self._frames), dtype=np.uint8),
                 frame_offsets=np.concatenate([[0], np.cumsum(sizes)]), **arrays)
        self.shards += 1
        self._columns = {name: [] for name in self._columns}
        self._frames = []

    def close(self) -> None:
        self.flush()
        meta = {**self.meta, "format": FORMAT, "rows": self.rows, "shards": self.shards,
                "stills": self.stills, "columns": sorted(self._columns),
                "frame": {"encoding": "jpeg", "scale": self.scale, "quality": self.quality}}
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Recording:
    """Reads a recording: its columns as whole arrays, its frames one at a time."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.shards = sorted(self.root.glob("shard_*.npz"))

    def __len__(self) -> int:
        return int(self.meta["rows"])

    def columns(self) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {}
        for shard in self.shards:
            with np.load(shard) as z:
                for name in z.files:
                    if not name.startswith("frame_"):
                        parts.setdefault(name, []).append(z[name])
        return {name: np.concatenate(values) for name, values in parts.items()}

    def frames(self) -> Iterator[Optional[np.ndarray]]:
        """Every row's frame (downscaled), in order; None where a row has no frame."""
        for shard in self.shards:
            with np.load(shard) as z:
                data, offsets = z["frame_bytes"], z["frame_offsets"]
            for a, b in zip(offsets[:-1], offsets[1:]):
                yield decode(data[a:b]) if b > a else None

    def stills(self) -> list[Path]:
        return sorted((self.root / "stills").glob("*.png"))
