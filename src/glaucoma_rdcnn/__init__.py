"""glaucoma-rdcnn: replication of the R-DCNN paper for joint OD/OC segmentation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("glaucoma-rdcnn")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
