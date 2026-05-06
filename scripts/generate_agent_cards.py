#!/usr/bin/env python3
"""
Agent Card Generator for NanoBrain Framework

Automatically generates mandatory Agent Cards for all agents
as required by the A2A protocol.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.card_schemas import (
    AgentCard, AgentCapabilities, Skill, IOSchema, ParameterSchema, UsageExample,
    InputMode, OutputMode, ParameterType, get_card_manager
)


def create_enhanced_collaborative_agent_card() -> AgentCard:
    """Create agent card for Enhanced Collaborative Agent."""
    
    # Define agent capabilities
    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=False,
        state_transition_history=True,
        multi_turn_conversation=True,
        context_retention=True,
        tool_usage=True,
        delegation=True,
        collaboration=True
    )
    
    # Define agent skills
    skills = [
        Skill(
            id="conversational_ai",
            name="Conversational AI",
            description="Advanced natural language understanding and generation for human-like interactions",
            tags=["conversation", "nlp", "communication"],
            examples=[
                "Engage in multi-turn conversations",
                "Understand context and intent",
                "Generate coherent responses"
            ],
            input_modes=[InputMode.TEXT],
            output_modes=[OutputMode.TEXT],
            complexity="advanced",
            dependencies=["language_model", "conversation_manager"]
        ),
        Skill(
            id="task_delegation",
            name="Task Delegation", 
            description="Ability to delegate tasks to other agents and coordinate distributed workflows",
            tags=["delegation", "coordination", "workflow"],
            examples=[
                "Delegate bioinformatics tasks to specialized agents",
                "Coordinate multi-agent workflows",
                "Monitor and manage task execution"
            ],
            input_modes=[InputMode.JSON, InputMode.TEXT],
            output_modes=[OutputMode.JSON, OutputMode.TEXT],
            complexity="advanced",
            dependencies=["agent_registry", "task_manager"]
        ),
        Skill(
            id="collaborative_reasoning",
            name="Collaborative Reasoning",
            description="Advanced reasoning capabilities that integrate inputs from multiple agents",
            tags=["reasoning", "collaboration", "integration"],
            examples=[
                "Synthesize results from multiple bioinformatics tools",
                "Resolve conflicts between agent recommendations",
                "Generate comprehensive analysis reports"
            ],
            input_modes=[InputMode.JSON, InputMode.TEXT],
            output_modes=[OutputMode.TEXT, OutputMode.STRUCTURED],
            complexity="expert",
            dependencies=["reasoning_engine", "conflict_resolution"]
        ),
        Skill(
            id="bioinformatics_coordination",
            name="Bioinformatics Workflow Coordination",
            description="Specialized coordination of bioinformatics analysis pipelines and tools",
            tags=["bioinformatics", "workflow", "pipeline"],
            examples=[
                "Coordinate viral protein analysis workflows",
                "Manage genome annotation pipelines",
                "Orchestrate comparative genomics studies"
            ],
            input_modes=[InputMode.JSON, InputMode.DATA],
            output_modes=[OutputMode.STRUCTURED, OutputMode.FILE],
            complexity="expert",
            dependencies=["bioinformatics_tools", "workflow_engine"]
        )
    ]
    
    # Input schema
    input_schema = IOSchema(
        format="object",
        description="Input schema for enhanced collaborative agent interactions",
        parameters=[
            ParameterSchema(
                name="message",
                type=ParameterType.STRING,
                description="Natural language message or instruction",
                required=True,
                example="Analyze the viral proteins in this genome and provide functional annotations"
            ),
            ParameterSchema(
                name="context",
                type=ParameterType.OBJECT,
                description="Context information for the conversation or task",
                required=False,
                example={"session_id": "abc123", "project": "alphavirus_study"}
            ),
            ParameterSchema(
                name="delegation_preferences",
                type=ParameterType.OBJECT,
                description="Preferences for task delegation and agent selection",
                required=False,
                example={"preferred_tools": ["BVBRCTool", "MMseqs2"], "timeout": 300}
            )
        ],
        examples=[
            {
                "message": "Analyze viral proteins in genome 123.45",
                "context": {"session_id": "abc123", "user": "researcher1"},
                "delegation_preferences": {"timeout": 600}
            }
        ],
        content_type="application/json"
    )
    
    # Output schema
    output_schema = IOSchema(
        format="object",
        description="Structured response from enhanced collaborative agent",
        parameters=[
            ParameterSchema(
                name="response",
                type=ParameterType.STRING,
                description="Natural language response to the user",
                required=True,
                example="I've analyzed the viral proteins and found 12 structural proteins including capsid and envelope proteins"
            ),
            ParameterSchema(
                name="analysis_results",
                type=ParameterType.OBJECT,
                description="Structured analysis results from delegated tasks",
                required=False,
                example={"protein_count": 12, "annotations": ["structural_protein", "capsid_protein"]}
            ),
            ParameterSchema(
                name="delegation_summary",
                type=ParameterType.OBJECT,
                description="Summary of tasks delegated to other agents",
                required=False,
                example={"tasks_delegated": 3, "agents_used": ["BVBRCTool", "MMseqs2"]}
            )
        ],
        examples=[
            {
                "response": "Analysis complete. Found 12 viral proteins with functional annotations.",
                "analysis_results": {"protein_count": 12, "functional_annotations": ["structural_protein", "capsid_protein"]},
                "delegation_summary": {"tasks_delegated": 2, "execution_time": "45s"}
            }
        ],
        content_type="application/json"
    )
    
    # Usage examples
    usage_examples = [
        UsageExample(
            name="Viral Genome Analysis",
            description="Comprehensive analysis of viral genome with protein annotation",
            input={
                "message": "Please analyze the proteins in Alphavirus genome 123.45 and provide functional annotations",
                "context": {"project": "alphavirus_comparative_study"}
            },
            expected_output={
                "response": "I've completed the analysis of Alphavirus genome 123.45. Found 12 proteins including structural and non-structural proteins with detailed functional annotations.",
                "analysis_results": {"total_proteins": 12, "structural_proteins": 5, "annotations": ["capsid_protein", "envelope_protein", "nsP1", "nsP2", "nsP3"]}
            },
            context="Viral genomics research with multi-agent coordination",
            tags=["viral_analysis", "protein_annotation", "collaboration"]
        ),
        UsageExample(
            name="Literature Integration", 
            description="Integrate bioinformatics results with literature research",
            input={
                "message": "Find recent literature about the proteins we just analyzed and summarize key findings",
                "context": {"previous_analysis": "protein_analysis_123"}
            },
            expected_output={
                "response": "Based on recent literature, the proteins show high similarity to known alphavirus structural proteins with conserved functional domains.",
                "analysis_results": {"literature_count": 25, "key_findings": ["conserved_domains", "structural_similarity", "functional_annotation"]}
            },
            context="Research synthesis and literature integration",
            tags=["literature_review", "research_synthesis", "integration"]
        )
    ]
    
    return AgentCard(
        name="EnhancedCollaborativeAgent",
        version="1.0.0",
        description="Advanced collaborative agent with delegation capabilities and bioinformatics expertise",
        purpose="Provides intelligent coordination of bioinformatics workflows through agent delegation, collaborative reasoning, and comprehensive analysis synthesis. Specializes in viral genomics and protein analysis with natural language interaction.",
        url="http://localhost:5001/agents/enhanced-collaborative",
        agent_type="collaborative",
        domain="bioinformatics",
        expertise_level="expert",
        capabilities=capabilities,
        skills=skills,
        supported_languages=["en"],
        default_input_modes=[InputMode.TEXT, InputMode.JSON],
        default_output_modes=[OutputMode.TEXT, OutputMode.JSON],
        input_schema=input_schema,
        output_schema=output_schema,
        conversation_patterns=["multi_turn", "task_oriented", "collaborative"],
        response_styles=["detailed", "scientific", "explanatory"],
        interaction_modes=["chat", "api", "workflow"],
        usage_examples=usage_examples,
        common_use_cases=[
            "Viral genome analysis coordination",
            "Multi-agent bioinformatics workflows", 
            "Research synthesis and integration",
            "Collaborative scientific analysis",
            "Literature-informed bioinformatics"
        ],
        limitations=[
            "Requires active agent ecosystem for delegation",
            "Performance depends on underlying tool availability",
            "Complex analyses may require extended processing time",
            "Limited to configured bioinformatics tools"
        ],
        max_context_length=8192,
        typical_response_time="10-120 seconds",
        concurrency_support=True,
        session_management=True,
        authentication={"schemes": ["none", "api_key"]},
        provider={"name": "NanoBrain", "organization": "Bioinformatics Research Lab"},
        documentation_url="https://nanobrain.readthedocs.io/agents/enhanced-collaborative",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        license="MIT"
    )


def create_collaborative_agent_card() -> AgentCard:
    """Create agent card for base Collaborative Agent."""
    
    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        multi_turn_conversation=True,
        context_retention=True,
        tool_usage=True,
        delegation=True,
        collaboration=True
    )
    
    skills = [
        Skill(
            id="collaborative_processing",
            name="Collaborative Processing",
            description="Coordinate tasks across multiple agents with result integration",
            tags=["collaboration", "coordination", "integration"],
            examples=[
                "Distribute bioinformatics tasks to specialized agents",
                "Integrate results from multiple analysis tools",
                "Manage workflow dependencies"
            ],
            input_modes=[InputMode.JSON, InputMode.TEXT],
            output_modes=[OutputMode.JSON, OutputMode.TEXT],
            complexity="advanced"
        ),
        Skill(
            id="task_orchestration",
            name="Task Orchestration",
            description="Orchestrate complex multi-step workflows across agent network",
            tags=["orchestration", "workflow", "management"],
            examples=[
                "Sequence protein analysis pipelines",
                "Coordinate parallel processing tasks",
                "Handle workflow error recovery"
            ],
            input_modes=[InputMode.JSON],
            output_modes=[OutputMode.JSON, OutputMode.STRUCTURED],
            complexity="advanced"
        )
    ]
    
    usage_examples = [
        UsageExample(
            name="Multi-Agent Workflow",
            description="Coordinate protein clustering and alignment across multiple tools",
            input={
                "task": "protein_analysis_workflow",
                "data": {"sequences": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"]},
                "agents": ["MMseqs2Tool", "MUSCLETool"]
            },
            expected_output={
                "status": "completed",
                "results": {"clusters": [{"id": 0, "members": ["seq1", "seq2"]}], "alignment": "MKTAYIAKQRQISFVK"},
                "workflow_summary": {"agents_used": 2, "total_time": "120s"}
            },
            context="Distributed bioinformatics analysis",
            tags=["workflow", "coordination", "bioinformatics"]
        )
    ]
    
    return AgentCard(
        name="CollaborativeAgent",
        version="1.0.0",
        description="Base collaborative agent for multi-agent task coordination",
        purpose="Coordinates tasks across multiple specialized agents, integrating results and managing workflow dependencies. Focuses on efficient distribution of computational tasks.",
        url="http://localhost:5001/agents/collaborative",
        agent_type="collaborative",
        domain="general",
        expertise_level="advanced",
        capabilities=capabilities,
        skills=skills,
        supported_languages=["en"],
        default_input_modes=[InputMode.JSON],
        default_output_modes=[OutputMode.JSON],
        conversation_patterns=["task_oriented"],
        response_styles=["structured", "concise"],
        interaction_modes=["api", "workflow"],
        usage_examples=usage_examples,
        common_use_cases=[
            "Multi-agent workflow coordination",
            "Distributed task processing",
            "Result integration and synthesis",
            "Workflow dependency management"
        ],
        limitations=[
            "Limited natural language capabilities",
            "Requires structured input formats",
            "Dependent on agent network availability"
        ],
        max_context_length=4096,
        typical_response_time="5-60 seconds",
        concurrency_support=True,
        session_management=False,
        authentication={"schemes": ["api_key"]},
        provider={"name": "NanoBrain", "organization": "Bioinformatics Research Lab"},
        documentation_url="https://nanobrain.readthedocs.io/agents/collaborative",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        license="MIT"
    )


def create_simple_specialized_agent_card() -> AgentCard:
    """Create agent card for Simple Specialized Agent."""
    
    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
        state_transition_history=False,
        multi_turn_conversation=False,
        context_retention=False,
        tool_usage=True,
        delegation=False,
        collaboration=False
    )
    
    skills = [
        Skill(
            id="specialized_processing",
            name="Specialized Processing",
            description="Focused processing for specific domain tasks",
            tags=["specialization", "focused", "domain_specific"],
            examples=[
                "Single-purpose bioinformatics analysis",
                "Specific data transformation tasks",
                "Targeted computational operations"
            ],
            input_modes=[InputMode.JSON, InputMode.DATA],
            output_modes=[OutputMode.JSON, OutputMode.DATA],
            complexity="intermediate"
        )
    ]
    
    usage_examples = [
        UsageExample(
            name="Single Task Processing",
            description="Execute single specialized bioinformatics task",
            input={
                "task": "sequence_alignment",
                "parameters": {"sequences": ["MKTAYIAKQRQISFVK", "MRVLQIAKQRQISFVK"], "method": "muscle"}
            },
            expected_output={
                "result": {"alignment": "MKTAYIAKQRQISFVK", "score": 0.85},
                "status": "completed"
            },
            context="Focused task execution",
            tags=["specialized", "single_task"]
        )
    ]
    
    return AgentCard(
        name="SimpleSpecializedAgent",
        version="1.0.0",
        description="Simple specialized agent for focused task execution",
        purpose="Executes specific, focused tasks without complex interaction patterns. Optimized for single-purpose operations with minimal overhead.",
        url="http://localhost:5001/agents/simple-specialized",
        agent_type="specialized",
        domain="general",
        expertise_level="intermediate",
        capabilities=capabilities,
        skills=skills,
        supported_languages=["en"],
        default_input_modes=[InputMode.JSON],
        default_output_modes=[OutputMode.JSON],
        conversation_patterns=["single_turn"],
        response_styles=["structured", "minimal"],
        interaction_modes=["api"],
        usage_examples=usage_examples,
        common_use_cases=[
            "Single-purpose task execution",
            "Simple data transformations",
            "Lightweight processing operations"
        ],
        limitations=[
            "No conversation memory",
            "Limited to single-turn interactions",
            "No collaboration capabilities"
        ],
        max_context_length=1024,
        typical_response_time="1-10 seconds",
        concurrency_support=False,
        session_management=False,
        authentication={"schemes": ["api_key"]},
        provider={"name": "NanoBrain", "organization": "Bioinformatics Research Lab"},
        documentation_url="https://nanobrain.readthedocs.io/agents/simple-specialized",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        license="MIT"
    )


def create_conversational_specialized_agent_card() -> AgentCard:
    """Create agent card for Conversational Specialized Agent."""
    
    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=False,
        state_transition_history=True,
        multi_turn_conversation=True,
        context_retention=True,
        tool_usage=True,
        delegation=False,
        collaboration=False
    )
    
    skills = [
        Skill(
            id="domain_conversation",
            name="Domain-Specific Conversation",
            description="Natural language interaction within specialized domain expertise",
            tags=["conversation", "domain_expertise", "natural_language"],
            examples=[
                "Discuss bioinformatics methodologies",
                "Explain analysis results in natural language",
                "Provide domain-specific guidance"
            ],
            input_modes=[InputMode.TEXT],
            output_modes=[OutputMode.TEXT],
            complexity="advanced"
        ),
        Skill(
            id="specialized_analysis",
            name="Specialized Analysis",
            description="Deep analysis capabilities within specific domain focus",
            tags=["analysis", "specialization", "expertise"],
            examples=[
                "Protein structure analysis",
                "Genomic variant interpretation",
                "Pathway analysis and annotation"
            ],
            input_modes=[InputMode.TEXT, InputMode.DATA],
            output_modes=[OutputMode.TEXT, OutputMode.STRUCTURED],
            complexity="expert"
        )
    ]
    
    usage_examples = [
        UsageExample(
            name="Expert Consultation",
            description="Provide expert analysis and explanation of bioinformatics results",
            input={
                "message": "Can you explain the significance of these protein clusters we found?",
                "data": {"clusters": [{"id": 1, "members": ["protein1", "protein2"]}]}
            },
            expected_output={
                "response": "These protein clusters suggest functional relationships. Cluster 1 contains proteins that likely share similar structural domains and biological functions.",
                "analysis": {"functional_prediction": "structural_similarity", "confidence": 0.85}
            },
            context="Expert consultation on analysis results",
            tags=["expert_analysis", "explanation", "consultation"]
        )
    ]
    
    return AgentCard(
        name="ConversationalSpecializedAgent",
        version="1.0.0",
        description="Conversational agent with deep domain specialization and expert knowledge",
        purpose="Combines natural language conversation capabilities with deep domain expertise. Provides expert-level analysis and explanations within specialized fields.",
        url="http://localhost:5001/agents/conversational-specialized",
        agent_type="conversational",
        domain="bioinformatics",
        expertise_level="expert",
        capabilities=capabilities,
        skills=skills,
        supported_languages=["en"],
        default_input_modes=[InputMode.TEXT],
        default_output_modes=[OutputMode.TEXT],
        conversation_patterns=["multi_turn", "consultative", "explanatory"],
        response_styles=["detailed", "educational", "expert"],
        interaction_modes=["chat", "consultation"],
        usage_examples=usage_examples,
        common_use_cases=[
            "Expert consultation and analysis",
            "Educational interactions about domain topics",
            "Detailed explanation of complex results",
            "Guided analysis workflows"
        ],
        limitations=[
            "Focused on specific domain expertise",
            "No delegation or collaboration features",
            "Requires domain-specific knowledge base"
        ],
        max_context_length=6144,
        typical_response_time="5-30 seconds",
        concurrency_support=False,
        session_management=True,
        authentication={"schemes": ["none", "session"]},
        provider={"name": "NanoBrain", "organization": "Bioinformatics Research Lab"},
        documentation_url="https://nanobrain.readthedocs.io/agents/conversational-specialized",
        created_date=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        license="MIT"
    )


def main():
    """Generate all agent cards for NanoBrain agents."""
    print("🤖 Generating Agent Cards for NanoBrain Framework...")
    
    # Initialize card manager
    card_manager = get_card_manager("cards")
    
    # Generate agent cards
    agents = [
        ("EnhancedCollaborativeAgent", create_enhanced_collaborative_agent_card),
        ("CollaborativeAgent", create_collaborative_agent_card),
        ("SimpleSpecializedAgent", create_simple_specialized_agent_card),
        ("ConversationalSpecializedAgent", create_conversational_specialized_agent_card)
    ]
    
    generated_cards = []
    
    for agent_name, card_creator in agents:
        print(f"  📋 Generating {agent_name} card...")
        try:
            agent_card = card_creator()
            card_manager.register_agent_card(agent_card)
            
            # Save in JSON format (A2A standard)
            json_path = card_manager.save_agent_card(agent_card, "json")
            
            generated_cards.append({
                "name": agent_name,
                "json_path": json_path
            })
            
            print(f"    ✅ {agent_name} card generated successfully")
            print(f"       JSON: {json_path}")
            
        except Exception as e:
            print(f"    ❌ Failed to generate {agent_name} card: {e}")
    
    print(f"\n🎉 Agent Card Generation Complete!")
    print(f"Generated {len(generated_cards)} agent cards:")
    
    for card in generated_cards:
        print(f"  • {card['name']}")
    
    print(f"\nCards saved to: {card_manager.cards_directory}")
    print("Agent cards are A2A protocol compliant and ready for agent discovery!")


if __name__ == "__main__":
    main()