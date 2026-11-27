"""pocketform-distill package. See README."""


def main() -> None:
    from pocketform_distill.cli import app  # imported lazily so `import pocketform_distill` stays light

    app()
