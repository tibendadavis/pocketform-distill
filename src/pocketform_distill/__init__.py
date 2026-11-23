"""pocketform-distill package. See README."""

from importlib.metadata import PackageNotFoundError, version


def main() -> None:
    try:
        v = version("pocketform-distill")
    except PackageNotFoundError:
        v = "unknown"
    print(f"pocketform-distill {v}")
    print("Pipeline stages are not implemented yet; see README.")
