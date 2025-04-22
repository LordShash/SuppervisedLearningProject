#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that the GUI can be closed without errors.

This script runs the GUI and prints a message when it's closed.
To test:
1. Run this script
2. Close the GUI using the X button
3. Check if the script exits cleanly without errors
"""

import os
import sys
import time

# Add the src directory to the path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

# Import the GUI module
from gui import main

if __name__ == "__main__":
    print("Starting GUI test...")
    print("Please close the GUI using the X button when it appears.")
    
    # Run the GUI
    main()
    
    print("GUI closed successfully!")
    print("If you see this message without any error traceback, the issue is fixed.")