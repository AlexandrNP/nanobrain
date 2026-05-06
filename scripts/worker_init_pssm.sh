#!/bin/bash
# PSSM Environment Worker Init - TESTED AND WORKING
# This script runs on Aurora compute nodes to set up Perl environment
# Last tested: 2025-12-02 - 7 second unpack time, all components verified
# UPDATED: Now preserves Parsl Python environment while adding PSSM tools

set -e

echo "=========================================="
echo "🔥 PSSM Worker Init on $(hostname)"
echo "📋 PBS_JOBID: ${PBS_JOBID}"
echo "🕐 Time: $(date)"
echo "=========================================="

# Configuration
SHARED_TARBALL="/lus/flare/projects/FoundEpidem/onarykov/nanobrain/envs/pssm-generation-env.tar.gz"
LOCAL_ENV="/tmp/pssm-env-${PBS_JOBID}"

# Step 0: Ensure Python environment with Parsl is available
# Add nanobrain-upd to PATH if not already active
NANOBRAIN_ENV="/home/onarykov/.conda/envs/nanobrain-upd"
if [ -d "$NANOBRAIN_ENV" ]; then
    echo "🔧 Adding nanobrain-upd environment to PATH (for Parsl)..."
    export PATH="$NANOBRAIN_ENV/bin:$PATH"
    export PYTHONPATH="/lus/flare/projects/FoundEpidem/onarykov/nanobrain:$PYTHONPATH"
    echo "✅ Python environment layered (Parsl available)"
    python3 --version
    which python3
else
    echo "⚠️  nanobrain-upd not found at $NANOBRAIN_ENV"
    echo "    Using system Python (Parsl must be installed system-wide)"
fi

# Step 1: Verify tarball exists
if [ ! -f "$SHARED_TARBALL" ]; then
    echo "❌ FATAL: Environment tarball not found"
    echo "   Expected: $SHARED_TARBALL"
    echo "   Run setup on login node first!"
    exit 1
fi

echo "✅ Found: $SHARED_TARBALL ($(du -h $SHARED_TARBALL | cut -f1))"

# Step 2: Unpack to node-local /tmp (fast I/O)
echo "📦 Unpacking to: $LOCAL_ENV"
mkdir -p "$LOCAL_ENV"

# Use pigz if available for parallel decompression
if command -v pigz &> /dev/null; then
    tar -I pigz -xf "$SHARED_TARBALL" -C "$LOCAL_ENV" 2>&1
else
    tar -xzf "$SHARED_TARBALL" -C "$LOCAL_ENV" 2>&1
fi

echo "✅ Unpacked ($(du -sh $LOCAL_ENV | cut -f1))"

# Step 3: DO NOT activate PSSM environment - just layer tools on PATH
# This preserves the Python environment with Parsl while adding PSSM tools
echo "🔧 Layering PSSM tools onto environment..."

# Step 4: Set environment variables
export PATH="$LOCAL_ENV/bin:$PATH"
export PERL5LIB="$LOCAL_ENV/lib/perl5/site_perl:$PERL5LIB"
export PSSM_ENV_ROOT="$LOCAL_ENV"
export PSSM_SCRIPT="$LOCAL_ENV/bin/fasta-cluster-pssm.pl"

# CRITICAL: Set default MAFFT threads (will be overridden by wrapper)
export MAFFT_THREADS=4

# CRITICAL: Unset MAFFT_BINARIES to avoid version mismatch warnings
unset MAFFT_BINARIES

# Step 5: Verify components (quick check)
echo "🔍 Environment verification:"
perl -e 'printf "  Perl: %vd\n", $^V'
echo "  psiblast: $(psiblast -version 2>&1 | head -1)"
echo "  mmseqs: $(mmseqs version 2>&1 | head -1)"

# Verify Perl modules (critical for execution)
perl -I"$PERL5LIB" -e 'use gjoseqlib; print "  ✅ gjoseqlib\n"' || {
    echo "  ❌ gjoseqlib FAILED"
    exit 1
}
perl -I"$PERL5LIB" -e 'use BlastInterface; print "  ✅ BlastInterface\n"' || {
    echo "  ❌ BlastInterface FAILED"
    exit 1
}
perl -I"$PERL5LIB" -e 'use SeedAware; print "  ✅ SeedAware\n"' || {
    echo "  ❌ SeedAware FAILED"
    exit 1
}

# Verify script exists
if [ ! -f "$PSSM_SCRIPT" ]; then
    echo "❌ FATAL: fasta-cluster-pssm.pl not found"
    exit 1
fi
echo "  ✅ Script: $PSSM_SCRIPT"

echo "=========================================="
echo "✅ PSSM Environment Ready"
echo "   Root: $PSSM_ENV_ROOT"
echo "   PERL5LIB: $PERL5LIB"
echo "   MAFFT_THREADS: $MAFFT_THREADS (default, override in wrapper)"
echo "=========================================="

# Cleanup on worker exit
trap "echo 'Cleaning up $LOCAL_ENV'; rm -rf '$LOCAL_ENV'" EXIT
