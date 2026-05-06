#!/usr/bin/env python3
"""
Focused smoke test for BV-BRC Data Acquisition Step with real BV-BRC calls.
"""
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.steps.data_acquisition_step import BVBRCDataAcquisitionStep

CONFIG_PATH = "nanobrain/library/workflows/viral_protein_analysis/config/DataAcquisitionStep/DataAcquisitionStep.yml"

async def main():
    # Point PATH to BV-BRC CLI for this run to ensure executables resolve
    os.environ["PATH"] = "/Applications/BV-BRC.app/deployment/bin:" + os.environ.get("PATH", "")

    # Create step directly from YAML path (required by framework)
    step = BVBRCDataAcquisitionStep.from_config(CONFIG_PATH)
    await step.initialize()

    # Test input per instruction: virus_species set to 'chikungunya'
    test_input = {
        "virus_species": "chikungunya",
        "analysis_parameters": {
            "use_cache": False
        }
    }

    print("Running BVBRCDataAcquisitionStep.process()...")
    result = await step.process(test_input)

    success = result.get("success", False)
    stats = result.get("statistics", {})
    fasta = result.get("annotated_fasta", "")

    print("Success:", success)
    print("Stats:", {k: stats.get(k) for k in [
        "unique_proteins_found", "sequences_retrieved", "annotations_retrieved", "fasta_entries"
    ]})
    print("Cache used:", result.get("cache_used"))
    print("Data source:", result.get("data_source"))
    print("FASTA length:", len(fasta or ""))

    if not success:
        print("Smoke test failed: step returned success=False")
        sys.exit(1)

    print("Smoke test passed")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())

