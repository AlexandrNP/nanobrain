#!/usr/bin/env python3
"""
Example script demonstrating the use of Langchain memory in the Agent class.
This script shows how to use both ConversationBufferMemory and ConversationBufferWindowMemory.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from the setup_paths.py file if it exists
setup_paths_file = project_root / "setup_paths.py"
if setup_paths_file.exists():
    from setup_paths import verify_paths
    verify_paths()

# Import necessary modules
from src.Agent import Agent
from src.ExecutorBase import ExecutorBase
from src.ConfigManager import ConfigManager

class SimpleExecutor(ExecutorBase):
    """Simple executor for testing."""
    def __init__(self):
        self.runnable_types = {"all"}
        self.energy_level = 1.0
        
    def can_execute(self, type_name):
        return True
        
    def execute(self, runnable):
        return runnable
        
    def recover_energy(self):
        self.energy_level = 1.0

async def test_buffer_memory():
    """Test ConversationBufferMemory in the Agent class."""
    print("\nTesting ConversationBufferMemory:")
    print("=================================")
    
    # Create an agent with ConversationBufferMemory
    agent = Agent(
        executor=SimpleExecutor(),
        use_buffer_window_memory=False,
        memory_key="chat_history",
        model_name="gpt-3.5-turbo",
        prompt_variables={
            "role_description": "assist with memory testing",
            "specific_instructions": "Remember previous interactions and refer to them when appropriate"
        }
    )
    
    # Add multiple interactions
    conversations = [
        "Hello, I'm testing your memory capabilities",
        "What's your favorite color?",
        "What did I ask you first?"
    ]
    
    # Process each conversation
    for i, message in enumerate(conversations):
        print(f"\nUser: {message}")
        response = await agent.process([message])
        print(f"Assistant: {response}")
        
        # Show memory after each interaction
        print("\nMemory state after interaction:")
        memory_vars = agent.langchain_memory.load_memory_variables({})
        print(f"- Memory size: {len(agent.memory)} message(s)")
        print(f"- Langchain memory has {len(memory_vars[agent.memory_key])} message(s)")

async def test_buffer_window_memory():
    """Test ConversationBufferWindowMemory in the Agent class."""
    print("\nTesting ConversationBufferWindowMemory:")
    print("=====================================")
    
    # Create an agent with ConversationBufferWindowMemory with a small window
    agent = Agent(
        executor=SimpleExecutor(),
        use_buffer_window_memory=True,
        memory_window_size=2,  # Only keep the last 2 interactions
        memory_key="chat_history",
        model_name="gpt-3.5-turbo",
        prompt_variables={
            "role_description": "assist with memory testing",
            "specific_instructions": "Remember previous interactions and refer to them when appropriate"
        }
    )
    
    # Add multiple interactions
    conversations = [
        "Hello, I'm testing your memory capabilities",
        "What's your favorite color?",
        "Can you remember what I asked first?",
        "What was the second question I asked you?",
        "Do you remember my first question now?"
    ]
    
    # Process each conversation
    for i, message in enumerate(conversations):
        print(f"\nUser: {message}")
        response = await agent.process([message])
        print(f"Assistant: {response}")
        
        # Show memory after each interaction
        print("\nMemory state after interaction:")
        memory_vars = agent.langchain_memory.load_memory_variables({})
        print(f"- Memory size: {len(agent.memory)} message(s)")
        print(f"- Langchain memory has {len(memory_vars[agent.memory_key])} message(s)")
        print(f"- Window size: {agent.memory_window_size} interactions")

async def test_shared_memory():
    """Test shared memory between agents."""
    print("\nTesting Shared Memory Between Agents:")
    print("===================================")
    
    # Create first agent
    agent1 = Agent(
        executor=SimpleExecutor(),
        use_buffer_window_memory=True,
        memory_window_size=5,
        use_shared_context=True,
        shared_context_key="shared_test",
        model_name="gpt-3.5-turbo",
        prompt_variables={
            "role_description": "assist with memory testing as Agent 1",
            "specific_instructions": "Remember you are Agent 1"
        }
    )
    
    # Process a message with the first agent
    print("\nInteracting with Agent 1:")
    user_message = "Hello Agent 1, please remember that my name is Alex"
    print(f"User: {user_message}")
    response = await agent1.process([user_message])
    print(f"Agent 1: {response}")
    
    # Show memory state of agent 1
    print("\nAgent 1 memory state:")
    agent1_memory = agent1.get_full_history()
    print(f"- Memory size: {len(agent1_memory)} message(s)")
    memory_vars = agent1.langchain_memory.load_memory_variables({})
    print(f"- Langchain memory has {len(memory_vars[agent1.memory_key])} message(s)")
    
    # Save to shared context
    agent1.save_to_shared_context("shared_test")
    print("- Saved to shared context 'shared_test'")
    
    # Create second agent with same shared context
    agent2 = Agent(
        executor=SimpleExecutor(),
        use_buffer_window_memory=True,
        memory_window_size=5,
        use_shared_context=True,
        shared_context_key="shared_test",
        model_name="gpt-3.5-turbo",
        prompt_variables={
            "role_description": "assist with memory testing as Agent 2",
            "specific_instructions": "Remember you are Agent 2"
        }
    )
    
    # Load from shared context
    agent2.load_from_shared_context("shared_test")
    print("\nAgent 2 created and loaded shared context.")
    
    # Show memory state of agent 2 after loading shared context
    agent2_memory = agent2.get_full_history()
    print(f"- Memory size after loading: {len(agent2_memory)} message(s)")
    memory_vars = agent2.langchain_memory.load_memory_variables({})
    print(f"- Langchain memory has {len(memory_vars[agent2.memory_key])} message(s)")
    
    # Interact with the second agent
    print("\nInteracting with Agent 2:")
    user_message = "Hello Agent 2, what is my name?"
    print(f"User: {user_message}")
    response = await agent2.process([user_message])
    print(f"Agent 2: {response}")
    
    # Show final memory state of both agents
    print("\nFinal memory state comparison:")
    agent1_mem = agent1.get_full_history()
    agent2_mem = agent2.get_full_history()
    
    print(f"Agent 1 memory size: {len(agent1_mem)} message(s)")
    print(f"Agent 2 memory size: {len(agent2_mem)} message(s)")
    
    # Check if Agent 2 contains all of Agent 1's memories
    if len(agent2_mem) > len(agent1_mem):
        print("Agent 2 has all of Agent 1's memories plus new interactions")
        matches = True
        for i in range(len(agent1_mem)):
            if agent1_mem[i]["content"] != agent2_mem[i]["content"]:
                matches = False
                print(f"Memory mismatch at position {i}")
                print(f"Agent 1: {agent1_mem[i]}")
                print(f"Agent 2: {agent2_mem[i]}")
        
        if matches:
            print("All shared memories match correctly")
    else:
        print("Error: Agent 2 should have more memories than Agent 1")

async def main():
    """Run all the examples."""
    print("Agent Memory Examples")
    print("====================")
    print("Testing how the Agent class uses Langchain memory mechanisms.")
    
    # Print if we're in testing mode
    testing_mode = os.environ.get('NANOBRAIN_TESTING', '0') == '1'
    if testing_mode:
        print("Running in TESTING mode with mock models.")
    else:
        print("Running with real LLM models. This will use your API credits.")
    
    # Test different memory mechanisms
    await test_buffer_memory()
    await test_buffer_window_memory()
    await test_shared_memory()

if __name__ == "__main__":
    # Set testing mode for this example
    os.environ['NANOBRAIN_TESTING'] = '1'
    
    # Run the examples
    asyncio.run(main()) 