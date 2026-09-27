#!/usr/bin/env python3

import sys
from .cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print("Program stoped by user.")
