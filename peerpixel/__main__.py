"""peerpixel - render for people whose machines cannot.

Everything is in `cli.py`. This file exists so that `python -m peerpixel` and
the `peerpixel` console script are the same program.
"""
from .cli import main

if __name__ == "__main__":
    main()
