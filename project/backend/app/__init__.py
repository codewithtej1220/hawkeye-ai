"""HawkEye backend package.

Windows Smart App Control can block the unsigned native library of `optree`, a package Keras
installs. PyTorch uses optree only if it finds it installed, and then fails to load with "An
Application Control policy has blocked this file". HawkEye does not need optree, so PyTorch is told
it is not installed. This must run before torch is imported; every backend entry point imports this
package first."""
import importlib.metadata as _metadata

_original_version = _metadata.version


def _version(name: str) -> str:
    if name.lower() == "optree":
        raise _metadata.PackageNotFoundError(name)
    return _original_version(name)


_metadata.version = _version
