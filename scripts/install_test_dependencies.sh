#!/bin/bash
# Chatbot Viral Integration - Test Dependencies Installation Script
# 
# This script installs all dependencies needed for comprehensive testing
# as outlined in the Issues Action Plan.

echo "🔧 Installing testing dependencies for Chatbot Viral Integration..."
echo "=================================================="

# Core testing framework
echo "📦 Installing pytest and async support..."
pip install pytest>=7.4.0
pip install pytest-asyncio>=0.21.0

# Production dependencies for full testing
echo "📦 Installing production dependencies..."
pip install cachetools>=5.3.0

# Enhanced testing utilities
echo "📦 Installing enhanced testing utilities..."
pip install pytest-html>=3.1.0  # HTML test reports
pip install pytest-cov>=4.1.0   # Coverage reporting

echo ""
echo "✅ Testing dependencies installed successfully!"
echo ""
echo "🧪 Quick Test Commands:"
echo "  Main Test Runner:    cd tests/chatbot_viral_integration && python3 test_runner.py"
echo "  Individual Tests:    cd tests/chatbot_viral_integration && pytest test_query_classification.py -v"
echo "  HTML Reports:        cd tests/chatbot_viral_integration && pytest --html=test_report.html"
echo "  Coverage Reports:    cd tests/chatbot_viral_integration && pytest --cov=. --cov-report=html"
echo ""
echo "📊 Verify Installation:"
echo "  python3 -c 'import pytest, cachetools; print(\"✅ All dependencies available\")'" 