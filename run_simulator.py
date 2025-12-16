import sys
import os

# Ensure the current directory is in the path so we can resolve 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulator.simulator import main

if __name__ == "__main__":
    main()
