#!/bin/bash
# Local Docker Build Verification Script

echo "🐳 Verifying Docker Environment..."

# 1. Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is NOT running."
    echo "   Please start Docker Desktop or OrbStack and try again."
    exit 1
fi

echo "✅ Docker is online."

# 2. Attempt Build
echo "🏗️  Building Docker Image (Dry Run)..."
docker build -t banjir-samarinda:test-build .

if [ $? -eq 0 ]; then
    echo "✅ Docker Build SUCCESS!"
    echo "   You can test the container with: docker run -p 8501:8501 banjir-samarinda:test-build"
    exit 0
else
    echo "❌ Docker Build FAILED."
    exit 1
fi
