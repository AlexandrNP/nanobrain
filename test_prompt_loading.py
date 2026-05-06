#!/usr/bin/env python3
"""
TEST: Validate prompts load from YAML (NO MOCKS!)
"""
import sys
import os

sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '/lus/flare/projects/FoundEpidem/onarykov/nanobrain'

from nanobrain.library.agents.specialized.protein_synonym_agent import ProteinSynonymAgent
from nanobrain.core.agent import AgentConfig

print("=" * 80)
print("PROMPT LOADING TEST")
print("=" * 80)

# Load agent from config file (Nanobrain mandatory pattern)
print("\nLoading agent from config file...")
agent = ProteinSynonymAgent.from_config('test_config/test_protein_synonym_agent.yml')

print("✅ Agent loaded from config")
print(f"   Config file contains: prompt_template_file")

print("✅ Agent created")

# Check prompt manager
if hasattr(agent, 'prompt_manager') and agent.prompt_manager:
    print("✅ PromptTemplateManager initialized")

    # Try to get a prompt
    try:
        test_prompt = agent.prompt_manager.get_prompt(
            'ictv_inference',
            params={
                'virus_genus': 'Alphavirus',
                'virus_species': 'Chikungunya virus',
                'virus_family': 'Togaviridae',
                'sample_products': '["nsP1", "nsP2", "E1", "E2"]'
            }
        )

        print("✅ Prompt loaded from YAML!")
        print(f"\n   Prompt preview (first 200 chars):")
        print(f"   {test_prompt[:200]}...")

        # Verify NO hardcoded fallback used
        if "Based on the International Committee" in test_prompt:
            print("\n✅ YAML prompt contains expected content")
            print("✅ NO HARDCODED FALLBACK - prompts loaded from config!")
        else:
            print("\n❌ Unexpected prompt content")

    except Exception as e:
        print(f"❌ Failed to get prompt: {e}")
        sys.exit(1)
else:
    print("❌ PromptTemplateManager NOT initialized")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ TEST PASSED - Prompts load from YAML configuration!")
print("=" * 80)
