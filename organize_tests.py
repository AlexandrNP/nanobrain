#!/usr/bin/env python3
"""
Script to organize test files from the root directory into appropriate test directories.

This script:
1. Identifies test files in the root directory
2. Moves unit tests to the test/ directory
3. Creates an integration_tests/ directory if it doesn't exist
4. Moves integration tests to the integration_tests/ directory
"""

import os
import shutil
import re
import sys

def main():
    """Main entry point for the script."""
    # Create integration_tests directory if it doesn't exist
    if not os.path.exists('integration_tests'):
        os.makedirs('integration_tests')
        print(f"Created directory: integration_tests")
    
    # Get all Python files in the root directory
    root_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.py')]
    
    # Identify test files
    test_files = [f for f in root_files if f.startswith('test_')]
    
    for file in test_files:
        # Determine if it's an integration test or a unit test
        with open(file, 'r') as f:
            content = f.read()
            
        is_integration_test = False
        
        # Check if file contains indicators of integration tests
        if (re.search(r'integration\s+test', content, re.IGNORECASE) or 
            'test_workflow' in file or 
            'test_step_creation' in file or
            'test_component_reuse' in file):
            is_integration_test = True
        
        # Destination directory
        dest_dir = 'integration_tests' if is_integration_test else 'test'
        
        # Move the file
        dest_path = os.path.join(dest_dir, file)
        if os.path.exists(dest_path):
            print(f"File already exists: {dest_path}")
            continue
        
        shutil.move(file, dest_path)
        print(f"Moved {file} to {dest_dir}/")

if __name__ == '__main__':
    # Check if this script is run from the root directory
    if not os.path.exists('test') or not os.path.isdir('test'):
        print("Error: This script must be run from the project root directory.")
        sys.exit(1)
    
    # Run the script
    main()
    print("Test organization complete!") 