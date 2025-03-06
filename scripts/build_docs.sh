#!/bin/bash

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Create docs/auto_generated directory if it doesn't exist
mkdir -p docs/auto_generated

# Run the documentation builder
python scripts/doc_builder.py

echo "Documentation built successfully!"
echo "You can find the generated documentation in docs/auto_generated/" 