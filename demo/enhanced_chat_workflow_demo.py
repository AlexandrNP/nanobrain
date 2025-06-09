#!/usr/bin/env python3
"""
Enhanced NanoBrain Chat Workflow Demo

An advanced demonstration of the NanoBrain framework featuring:
- Persistent conversation history
- Performance metrics dashboard
- Enhanced CLI interface with more commands
- Multi-turn context management
- Real-time statistics
- Export/import capabilities
- A2A (Agent-to-Agent) protocol support
- MCP (Model Context Protocol) support
- Multi-agent collaboration capabilities

Architecture:
CLI Input → User Input DataUnit → Agent Input DataUnit → Enhanced Agent Step → Agent Output DataUnit → CLI Output
                                                    ↓
                                            Conversation History
                                            Performance Metrics
                                            A2A Agent Collaboration
                                            MCP Tool Integration
"""

import asyncio
import sys
import os
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_unit import DataUnitMemory, DataUnitConfig, DataUnitType
from core.trigger import DataUpdatedTrigger, TriggerConfig, TriggerType
from core.link import DirectLink, LinkConfig, LinkType
from core.step import Step, StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import LocalExecutor, ExecutorConfig
from core.a2a_support import A2ASupportMixin, with_a2a_support
from core.mcp_support import MCPSupportMixin
# from config.component_factory import ComponentFactory, get_factory


class EnhancedCollaborativeAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """
    Enhanced conversational agent with A2A and MCP protocol support.
    
    This agent can:
    - Use MCP tools for structured operations
    - Collaborate with A2A agents for specialized tasks
    - Maintain conversation context and history
    - Provide performance metrics and monitoring
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.collaboration_count = 0
        self.tool_usage_count = 0
        self.delegation_rules = kwargs.get('delegation_rules', [])
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Enhanced process method with A2A delegation and MCP tool usage."""
        # Check if we should delegate to an A2A agent
        if self.a2a_enabled and self.a2a_agents:
            delegation_result = await self._check_for_delegation(input_text)
            if delegation_result:
                self.collaboration_count += 1
                return delegation_result
        
        # Check if we should use MCP tools
        if self.mcp_enabled and self.mcp_tools:
            tool_result = await self._check_for_tool_usage(input_text)
            if tool_result:
                self.tool_usage_count += 1
                return tool_result
        
        # Fall back to normal processing
        return await super().process(input_text, **kwargs)
    
    async def _check_for_delegation(self, input_text: str) -> Optional[str]:
        """Check if the input should be delegated to an A2A agent."""
        input_lower = input_text.lower()
        
        # Check delegation rules
        for rule in self.delegation_rules:
            keywords = rule.get('keywords', [])
            agent_name = rule.get('agent')
            
            if any(keyword in input_lower for keyword in keywords):
                if agent_name in self.a2a_agents:
                    try:
                        # Log delegation
                        self.nb_logger.info(f"Delegating to A2A agent: {agent_name}",
                                          rule_description=rule.get('description', ''),
                                          collaboration_count=self.collaboration_count + 1)
                        
                        # Call A2A agent
                        result = await self.call_a2a_agent(agent_name, input_text)
                        
                        # Wrap result with context
                        return f"🤝 Collaborated with {agent_name}:\n\n{result}"
                        
                    except Exception as e:
                        self.nb_logger.error(f"A2A delegation failed: {e}")
                        # Continue with normal processing
                        break
        
        return None
    
    async def _check_for_tool_usage(self, input_text: str) -> Optional[str]:
        """Check if the input should use MCP tools."""
        input_lower = input_text.lower()
        
        # Simple keyword-based tool detection
        tool_keywords = {
            'calculator': ['calculate', 'math', 'compute', 'add', 'subtract', 'multiply', 'divide'],
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'file': ['file', 'read', 'write', 'save', 'load']
        }
        
        for tool_name, keywords in tool_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                if tool_name in self.mcp_tools:
                    try:
                        # Log tool usage
                        self.nb_logger.info(f"Using MCP tool: {tool_name}",
                                          tool_usage_count=self.tool_usage_count + 1)
                        
                        # Use MCP tool (simplified - would need actual tool calling logic)
                        result = f"🔧 Used {tool_name} tool to process: {input_text}"
                        
                        return result
                        
                    except Exception as e:
                        self.nb_logger.error(f"MCP tool usage failed: {e}")
                        # Continue with normal processing
                        break
        
        return None
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status including A2A and MCP information."""
        status = {
            'agent_name': self.config.name,
            'collaboration_count': self.collaboration_count,
            'tool_usage_count': self.tool_usage_count
        }
        
        # Add A2A status
        if hasattr(self, 'get_a2a_status'):
            status['a2a'] = self.get_a2a_status()
        
        # Add MCP status
        if hasattr(self, 'get_mcp_status'):
            status['mcp'] = self.get_mcp_status()
        
        return status


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    timestamp: datetime
    user_input: str
    agent_response: str
    response_time_ms: float
    conversation_id: str
    message_id: int


@dataclass
class PerformanceMetrics:
    """Performance metrics for the chat workflow."""
    total_conversations: int = 0
    total_messages: int = 0
    average_response_time_ms: float = 0.0
    total_response_time_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
    messages_per_minute: float = 0.0
    start_time: datetime = None


class ConversationHistoryManager:
    """Manages persistent conversation history using SQLite."""
    
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database for conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                response_time_ms REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conversation_id 
            ON conversations(conversation_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    async def save_message(self, message: ConversationMessage):
        """Save a conversation message to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (conversation_id, message_id, timestamp, user_input, agent_response, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            message.conversation_id,
            message.message_id,
            message.timestamp.isoformat(),
            message.user_input,
            message.agent_response,
            message.response_time_ms
        ))
        
        conn.commit()
        conn.close()
    
    async def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Retrieve conversation history for a specific conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_id, message_id, timestamp, user_input, agent_response, response_time_ms
            FROM conversations 
            WHERE conversation_id = ?
            ORDER BY message_id DESC
            LIMIT ?
        ''', (conversation_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append(ConversationMessage(
                conversation_id=row[0],
                message_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                user_input=row[3],
                agent_response=row[4],
                response_time_ms=row[5]
            ))
        
        return list(reversed(messages))  # Return in chronological order
    
    async def get_recent_conversations(self, hours: int = 24) -> List[str]:
        """Get list of recent conversation IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT DISTINCT conversation_id
            FROM conversations 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (since_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    async def export_conversations(self, output_file: str):
        """Export all conversations to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_id, message_id, timestamp, user_input, agent_response, response_time_ms
            FROM conversations 
            ORDER BY conversation_id, message_id
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        conversations = defaultdict(list)
        for row in rows:
            conversations[row[0]].append({
                'message_id': row[1],
                'timestamp': row[2],
                'user_input': row[3],
                'agent_response': row[4],
                'response_time_ms': row[5]
            })
        
        with open(output_file, 'w') as f:
            json.dump(dict(conversations), f, indent=2)


class PerformanceTracker:
    """Tracks and manages performance metrics."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics(start_time=datetime.now())
        self.response_times = []
        self.message_timestamps = []
    
    def record_message(self, response_time_ms: float, error: bool = False):
        """Record a message interaction."""
        self.metrics.total_messages += 1
        self.metrics.total_response_time_ms += response_time_ms
        self.metrics.average_response_time_ms = (
            self.metrics.total_response_time_ms / self.metrics.total_messages
        )
        
        if error:
            self.metrics.error_count += 1
        
        # Track recent response times (last 100)
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        # Track message timestamps for rate calculation
        now = datetime.now()
        self.message_timestamps.append(now)
        
        # Keep only last hour of timestamps
        cutoff = now - timedelta(hours=1)
        self.message_timestamps = [ts for ts in self.message_timestamps if ts > cutoff]
        
        # Calculate messages per minute
        if len(self.message_timestamps) > 1:
            time_span = (self.message_timestamps[-1] - self.message_timestamps[0]).total_seconds()
            if time_span > 0:
                self.metrics.messages_per_minute = len(self.message_timestamps) / (time_span / 60)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        self.metrics.uptime_seconds = (datetime.now() - self.metrics.start_time).total_seconds()
        return self.metrics
    
    def get_recent_response_times(self) -> List[float]:
        """Get recent response times for analysis."""
        return self.response_times.copy()


class EnhancedConversationalAgentStep(Step):
    """
    Enhanced step wrapper for EnhancedCollaborativeAgent with metrics, history, and protocol support.
    """
    
    def __init__(self, config: StepConfig, agent: EnhancedCollaborativeAgent, 
                 history_manager: ConversationHistoryManager,
                 performance_tracker: PerformanceTracker):
        super().__init__(config)
        self.agent = agent
        self.history_manager = history_manager
        self.performance_tracker = performance_tracker
        self.conversation_count = 0
        self.current_conversation_id = f"conv_{int(time.time())}"
        self.message_id = 0
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input through the conversational agent with enhanced tracking.
        """
        user_input = inputs.get('user_input', '')
        if not user_input or user_input.strip() == '':
            return {'agent_response': ''}
        
        start_time = time.time()
        self.message_id += 1
        
        try:
            # Process through conversational agent
            response = await self.agent.process(user_input)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record performance metrics
            self.performance_tracker.record_message(response_time_ms, error=False)
            
            # Save to conversation history
            message = ConversationMessage(
                timestamp=datetime.now(),
                user_input=user_input,
                agent_response=response or 'I apologize, but I could not generate a response.',
                response_time_ms=response_time_ms,
                conversation_id=self.current_conversation_id,
                message_id=self.message_id
            )
            
            await self.history_manager.save_message(message)
            
            self.nb_logger.info(f"Processed message #{self.message_id}", 
                           user_input_length=len(user_input),
                           response_length=len(response) if response else 0,
                           response_time_ms=response_time_ms,
                           conversation_id=self.current_conversation_id)
            
            return {'agent_response': response or 'I apologize, but I could not generate a response.'}
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.performance_tracker.record_message(response_time_ms, error=True)
            
            self.nb_logger.error(f"Error processing user input: {e}", 
                            error_type=type(e).__name__,
                            response_time_ms=response_time_ms)
            
            error_response = f'Sorry, I encountered an error: {str(e)}'
            
            # Still save error to history
            message = ConversationMessage(
                timestamp=datetime.now(),
                user_input=user_input,
                agent_response=error_response,
                response_time_ms=response_time_ms,
                conversation_id=self.current_conversation_id,
                message_id=self.message_id
            )
            
            await self.history_manager.save_message(message)
            
            return {'agent_response': error_response}
    
    def start_new_conversation(self):
        """Start a new conversation session."""
        import time
        self.current_conversation_id = f"conv_{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
        self.message_id = 0
        self.conversation_count += 1


class EnhancedCLIInterface:
    """
    Enhanced Command Line Interface with more features and better UX.
    """
    
    def __init__(self, input_data_unit: DataUnitMemory, output_data_unit: DataUnitMemory,
                 history_manager: ConversationHistoryManager,
                 performance_tracker: PerformanceTracker,
                 agent_step: EnhancedConversationalAgentStep):
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
        self.history_manager = history_manager
        self.performance_tracker = performance_tracker
        self.agent_step = agent_step
        self.running = False
        self.input_thread = None
        self.show_metrics = False
        self.metrics_thread = None
        self.response_queue = asyncio.Queue()
        self.last_message_id = 0
        self.event_loop = None  # Store event loop reference
        
        # Response statistics tracking (like Parsl demo)
        self.response_stats = {
            'total_requests': 0,
            'total_responses': 0,
            'total_processing_time': 0.0,
            'error_count': 0
        }
        
        # Coordination for input/output timing
        self.waiting_for_response = False
        self.response_received_event = None
        
    async def start(self, show_welcome=True):
        """Start the enhanced CLI interface."""
        self.running = True
        
        # Store event loop reference for use in threads
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError as e:
            print(f"❌ Failed to get event loop: {e}")
            raise
        
        # Initialize response coordination event
        self.response_received_event = asyncio.Event()
        
        # Note: Output monitoring is handled by trigger callbacks, not polling
        # This eliminates duplicate response display
        
        if show_welcome:
            self._show_welcome()
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        # Start metrics display thread if enabled
        if self.show_metrics:
            self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self.metrics_thread.start()
    
    async def start_for_testing(self):
        """Start the CLI interface for testing without welcome message or input thread."""
        self.running = True
        
        # Store event loop reference for use in threads
        self.event_loop = asyncio.get_running_loop()
        
        # Note: Output monitoring is handled by trigger callbacks, not polling
        # This eliminates duplicate response display
        
        # Don't start input thread or show welcome for testing
        
    async def stop(self):
        """Stop the CLI interface."""
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        if self.metrics_thread:
            self.metrics_thread.join(timeout=1.0)
    
    def _show_welcome(self):
        """Show enhanced welcome message."""
        print("🧠 Enhanced NanoBrain Chat Workflow Demo")
        print("=" * 60)
        print("🚀 Features: History • Metrics • Multi-turn Context • Export • A2A • MCP")
        print("📝 Type your messages below or use commands:")
        print("   /help     - Show all commands")
        print("   /quit     - Exit the chat")
        print("   /new      - Start new conversation")
        print("   /history  - Show conversation history")
        print("   /stats    - Show performance statistics")
        print("   /status   - Show A2A and MCP status")
        print("   /export   - Export conversations")
        print("=" * 60)
    
    def _input_loop(self):
        """Enhanced input loop with command processing."""
        while self.running:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command_sync(user_input)
                    continue
                
                # Handle regular chat input
                self._handle_user_input_sync(user_input)
                
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Input error: {e}")
    
    def _handle_command_sync(self, command: str):
        """Handle commands synchronously from input thread."""
        try:
            if self.event_loop and not self.event_loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self._handle_command(command),
                    self.event_loop
                )
                # Wait for command to complete with timeout
                future.result(timeout=5.0)
            else:
                print("❌ Event loop not available")
        except Exception as e:
            print(f"❌ Command error: {e}")
    
    def _handle_user_input_sync(self, user_input: str):
        """Handle user input synchronously from input thread."""
        try:
            if self.event_loop and not self.event_loop.is_closed():
                # Track the request
                self.response_stats['total_requests'] += 1
                
                # Set waiting flag and clear the event
                self.waiting_for_response = True
                if self.response_received_event:
                    # Clear the event - this is not a coroutine
                    self.response_received_event.clear()
                
                future = asyncio.run_coroutine_threadsafe(
                    self.input_data_unit.set({'user_input': user_input}),
                    self.event_loop
                )
                # Don't wait for this to complete to avoid blocking input
                
                # Wait for response with timeout
                if self.response_received_event:
                    try:
                        wait_future = asyncio.run_coroutine_threadsafe(
                            asyncio.wait_for(self.response_received_event.wait(), timeout=10.0),
                            self.event_loop
                        )
                        wait_future.result(timeout=11.0)  # Slightly longer timeout for the future itself
                    except (asyncio.TimeoutError, Exception):
                        # Continue if timeout or error - don't block indefinitely
                        pass
                    finally:
                        self.waiting_for_response = False
            else:
                print("❌ Event loop not available")
        except Exception as e:
            print(f"❌ Input processing error: {e}")
            self.waiting_for_response = False
    
    async def _handle_command(self, command: str):
        """Handle CLI commands."""
        cmd = command.lower().strip()
        
        if cmd in ['/quit', '/exit', '/bye']:
            print("👋 Goodbye!")
            self.running = False
            
        elif cmd == '/help':
            self._show_help()
            
        elif cmd == '/new':
            self.agent_step.start_new_conversation()
            print("🆕 Started new conversation")
            
        elif cmd == '/history':
            await self._show_history()
            
        elif cmd == '/stats':
            await self._show_stats()
            
        elif cmd == '/status':
            await self._show_protocol_status()
            
        elif cmd == '/export':
            await self._export_conversations()
            
        elif cmd == '/metrics':
            self.show_metrics = not self.show_metrics
            if self.show_metrics:
                print("📊 Real-time metrics enabled")
                if not self.metrics_thread or not self.metrics_thread.is_alive():
                    self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
                    self.metrics_thread.start()
            else:
                print("📊 Real-time metrics disabled")
                
        elif cmd == '/recent':
            await self._show_recent_conversations()
            
        else:
            print(f"❓ Unknown command: {command}")
            print("   Type /help for available commands")
    
    async def _show_history(self):
        """Show conversation history."""
        try:
            history = await self.history_manager.get_conversation_history(
                self.agent_step.current_conversation_id, limit=10
            )
            
            if not history:
                print("📝 No conversation history found")
                return
            
            print(f"\n📝 Conversation History (Last 10 messages)")
            print(f"   Conversation ID: {self.agent_step.current_conversation_id}")
            print("-" * 50)
            
            for msg in history:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                print(f"[{timestamp}] 👤: {msg.user_input}")
                print(f"[{timestamp}] 🤖: {msg.agent_response}")
                print(f"           ⏱️  {msg.response_time_ms:.1f}ms")
                print()
                
        except Exception as e:
            print(f"❌ Error showing history: {e}")
    
    async def _show_stats(self):
        """Show performance statistics."""
        try:
            metrics = self.performance_tracker.get_current_metrics()
            recent_times = self.performance_tracker.get_recent_response_times()
            
            print(f"\n📊 Performance Statistics")
            print("-" * 40)
            print(f"🔢 Total Messages: {metrics.total_messages}")
            print(f"💬 Total Conversations: {self.agent_step.conversation_count}")
            print(f"⏱️  Average Response Time: {metrics.average_response_time_ms:.1f}ms")
            print(f"🚀 Messages/Minute: {metrics.messages_per_minute:.1f}")
            print(f"❌ Error Count: {metrics.error_count}")
            print(f"⏰ Uptime: {metrics.uptime_seconds/60:.1f} minutes")
            
            if recent_times:
                print(f"📈 Recent Response Times:")
                print(f"   Min: {min(recent_times):.1f}ms")
                print(f"   Max: {max(recent_times):.1f}ms")
                print(f"   Median: {sorted(recent_times)[len(recent_times)//2]:.1f}ms")
            
        except Exception as e:
            print(f"❌ Error showing stats: {e}")
    
    async def _show_protocol_status(self):
        """Show A2A and MCP protocol status."""
        try:
            print(f"\n🔗 Protocol Status")
            print("-" * 40)
            
            # Get enhanced status from agent
            if hasattr(self.agent_step.agent, 'get_enhanced_status'):
                status = self.agent_step.agent.get_enhanced_status()
                
                print(f"🤖 Agent: {status.get('agent_name', 'Unknown')}")
                print(f"🤝 A2A Collaborations: {status.get('collaboration_count', 0)}")
                print(f"🔧 MCP Tool Usage: {status.get('tool_usage_count', 0)}")
                
                # A2A Status
                a2a_status = status.get('a2a', {})
                if a2a_status:
                    print(f"\n📡 A2A Protocol:")
                    print(f"   Enabled: {a2a_status.get('enabled', False)}")
                    print(f"   Client Initialized: {a2a_status.get('client_initialized', False)}")
                    print(f"   Available Agents: {a2a_status.get('total_agents', 0)}")
                    
                    agents = a2a_status.get('agents', {})
                    if agents:
                        print(f"   Configured Agents:")
                        for name, info in agents.items():
                            print(f"     - {name}: {info.get('description', 'No description')}")
                            print(f"       Skills: {info.get('skills_count', 0)}")
                else:
                    print(f"\n📡 A2A Protocol: Not configured")
                
                # MCP Status
                mcp_status = status.get('mcp', {})
                if mcp_status:
                    print(f"\n🔧 MCP Protocol:")
                    print(f"   Enabled: {mcp_status.get('enabled', False)}")
                    print(f"   Client Initialized: {mcp_status.get('client_initialized', False)}")
                    print(f"   Available Tools: {mcp_status.get('total_tools', 0)}")
                    
                    tools = mcp_status.get('tools', {})
                    if tools:
                        print(f"   Configured Tools:")
                        for name, info in tools.items():
                            print(f"     - {name}: {info.get('description', 'No description')}")
                else:
                    print(f"\n🔧 MCP Protocol: Not configured")
            else:
                print("⚠️  Enhanced status not available")
                
        except Exception as e:
            print(f"❌ Error showing protocol status: {e}")
    
    async def _export_conversations(self):
        """Export conversations to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{timestamp}.json"
            
            await self.history_manager.export_conversations(filename)
            print(f"💾 Conversations exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting conversations: {e}")
    
    async def _show_recent_conversations(self):
        """Show recent conversation IDs."""
        try:
            recent = await self.history_manager.get_recent_conversations(hours=24)
            
            if not recent:
                print("📝 No recent conversations found")
                return
            
            print(f"\n📝 Recent Conversations (Last 24 hours)")
            print("-" * 40)
            for conv_id in recent[:10]:  # Show last 10
                print(f"   {conv_id}")
                
        except Exception as e:
            print(f"❌ Error showing recent conversations: {e}")
    
    def _metrics_loop(self):
        """Real-time metrics display loop."""
        while self.running and self.show_metrics:
            try:
                time.sleep(5)  # Update every 5 seconds
                if not self.running:
                    break
                    
                metrics = self.performance_tracker.get_current_metrics()
                
                # Clear previous metrics display (simple approach)
                print(f"\r📊 Live: {metrics.total_messages} msgs | "
                      f"{metrics.average_response_time_ms:.0f}ms avg | "
                      f"{metrics.messages_per_minute:.1f}/min", end="", flush=True)
                      
            except Exception:
                break
    
    # Note: Output monitoring is now handled by trigger callbacks only
    # This eliminates duplicate response display and follows NanoBrain architecture
    
    async def _handle_response(self, response_data: Dict[str, Any]):
        """Handle response from agents following Parsl demo pattern."""
        try:
            # Handle the response data structure
            response_text = ""
            metadata = response_data.get('metadata', {})
            
            # Check for agent_response field (enhanced demo format)
            if 'agent_response' in response_data:
                response_text = response_data['agent_response']
            # Check for response field (generic format)
            elif 'response' in response_data:
                response_text = response_data['response']
            # Handle direct string response
            elif isinstance(response_data, str):
                response_text = response_data
            else:
                response_text = str(response_data)
            
            # Clean up the response text if it contains error info
            if response_text.startswith('Error: Error code:'):
                response_text = "I apologize, but I encountered an error processing your request. Please try again."
            
            # Display the bot response
            if response_text and response_text.strip():
                if self.show_metrics:
                    print()  # New line after metrics
                
                print(f"\n🤖 Bot: {response_text}")
                
                # Show processing info if available and significant
                processing_time = metadata.get('processing_time', 0)
                if processing_time and processing_time > 1.0:
                    agent_id = metadata.get('agent_id', 'assistant')
                    print(f"     (processed in {processing_time:.2f}s by {agent_id})")
                
                # Add spacing after response
                print()
                
                # Update response statistics
                if hasattr(self, 'response_stats'):
                    self.response_stats['total_responses'] += 1
                    self.response_stats['total_processing_time'] += processing_time
            
            # Check for errors
            if response_data.get('error'):
                if hasattr(self, 'response_stats'):
                    self.response_stats['error_count'] += 1
                print(f"⚠️  Error: {response_data['error']}")
            
            # Signal that response has been displayed
            if self.waiting_for_response and self.response_received_event:
                self.response_received_event.set()
            
        except Exception as e:
            print(f"⚠️  Error displaying response: {e}")
            # Signal completion even on error
            if self.waiting_for_response and self.response_received_event:
                self.response_received_event.set()
    
    async def _on_output_received(self, data: Dict[str, Any]):
        """Handle output from the agent (legacy method for compatibility)."""
        await self._handle_response(data)
    
    def _show_help(self):
        """Show comprehensive help information."""
        print("\n📋 Enhanced Chat Commands:")
        print("  /help      - Show this help message")
        print("  /quit      - Exit the chat")
        print("  /new       - Start a new conversation")
        print("  /history   - Show current conversation history")
        print("  /stats     - Show performance statistics")
        print("  /status    - Show A2A and MCP protocol status")
        print("  /export    - Export all conversations to JSON")
        print("  /metrics   - Toggle real-time metrics display")
        print("  /recent    - Show recent conversation IDs")
        print("\n💡 Enhanced Features:")
        print("  - Persistent conversation history")
        print("  - Real-time performance metrics")
        print("  - Multi-turn context management")
        print("  - Conversation export/import")
        print("  - Response time tracking")
        print("  - Error rate monitoring")
        print("  - A2A agent-to-agent collaboration")
        print("  - MCP tool integration")


class EnhancedChatWorkflow:
    """
    Enhanced chat workflow orchestrator with additional features.
    """
    
    def __init__(self):
        # self.factory = get_factory()
        self.components = {}
        self.cli = None
        self.executor = None
        self.history_manager = ConversationHistoryManager()
        self.performance_tracker = PerformanceTracker()
        
    async def setup(self):
        """Set up the enhanced chat workflow."""
        print("🔧 Setting up Enhanced NanoBrain Chat Workflow...")
        
        # 1. Create executor
        print("   Creating executor...")
        executor_config = ExecutorConfig(
            executor_type="local",
            max_workers=4,  # Increased for better performance
            timeout=30.0
        )
        self.executor = LocalExecutor(executor_config)
        await self.executor.initialize()
        
        # 2. Create data units
        print("   Creating data units...")
        
        # User input data unit
        user_input_config = DataUnitConfig(
            data_type=DataUnitType.MEMORY,
            persistent=False,
            cache_size=200  # Increased cache
        )
        user_input_du = DataUnitMemory(user_input_config, name="user_input")
        await user_input_du.initialize()
        self.components['user_input_du'] = user_input_du
        
        # Agent input data unit
        agent_input_config = DataUnitConfig(
            data_type=DataUnitType.MEMORY,
            persistent=False,
            cache_size=200
        )
        agent_input_du = DataUnitMemory(agent_input_config, name="agent_input")
        await agent_input_du.initialize()
        self.components['agent_input_du'] = agent_input_du
        
        # Agent output data unit
        agent_output_config = DataUnitConfig(
            data_type=DataUnitType.MEMORY,
            persistent=False,
            cache_size=200
        )
        agent_output_du = DataUnitMemory(agent_output_config, name="agent_output")
        await agent_output_du.initialize()
        self.components['agent_output_du'] = agent_output_du
        
        # 3. Create enhanced collaborative agent
        print("   Creating enhanced collaborative agent with A2A and MCP support...")
        agent_config = AgentConfig(
            name="EnhancedCollaborativeAssistant",
            description="Enhanced conversational assistant with A2A and MCP protocol support",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500,  # Increased for longer responses
            system_prompt="""You are an advanced AI assistant with enhanced capabilities including multi-agent collaboration and tool integration. You maintain conversation context, provide detailed and helpful responses, and can engage in complex multi-turn conversations.

Key capabilities:
- Maintain conversation context across multiple turns
- Collaborate with specialized A2A agents for complex tasks
- Use MCP tools for structured operations
- Provide detailed explanations with examples
- Ask clarifying questions when needed
- Offer suggestions and follow-up topics
- Adapt your communication style to the user's needs
- Remember previous topics in the conversation
- Provide structured responses when appropriate

Multi-agent collaboration:
- Delegate travel-related tasks to travel agents
- Delegate coding tasks to code agents
- Delegate data analysis to data agents
- Use appropriate tools for calculations, file operations, etc.

Guidelines:
- Be conversational, helpful, and engaging
- Show enthusiasm for learning and helping
- Provide accurate and well-reasoned responses
- Use examples and analogies when helpful
- Acknowledge when you don't know something
- Offer to explore topics in more depth
- Be transparent about when you're collaborating with other agents or using tools""",
            auto_initialize=False,
            debug_mode=True,
            enable_logging=True,
            log_conversations=True,
            log_tool_calls=True
        )
        
        # Create delegation rules for A2A
        delegation_rules = [
            {
                'keywords': ['flight', 'hotel', 'travel', 'trip', 'vacation', 'booking'],
                'agent': 'travel_agent',
                'description': 'Delegate travel-related requests to travel specialist'
            },
            {
                'keywords': ['code', 'program', 'function', 'algorithm', 'debug', 'python', 'javascript'],
                'agent': 'code_agent',
                'description': 'Delegate programming tasks to code specialist'
            },
            {
                'keywords': ['data', 'analyze', 'chart', 'graph', 'statistics', 'visualization'],
                'agent': 'data_agent',
                'description': 'Delegate data analysis tasks to data specialist'
            }
        ]
        
        agent = EnhancedCollaborativeAgent(
            agent_config,
            a2a_enabled=True,
            a2a_config_path="config/a2a_config.yaml",
            mcp_enabled=True,
            mcp_config_path="config/mcp_config.yaml",
            delegation_rules=delegation_rules
        )
        
        await agent.initialize()
        self.components['agent'] = agent
        
        # 4. Create enhanced agent step
        print("   Creating enhanced agent step...")
        step_config = StepConfig(
            name="enhanced_chat_agent_step",
            description="Enhanced conversational agent step with history and metrics",
            debug_mode=True,
            enable_logging=True
        )
        
        agent_step = EnhancedConversationalAgentStep(
            step_config, agent, self.history_manager, self.performance_tracker
        )
        
        # Register data units with step
        agent_step.register_input_data_unit('user_input', agent_input_du)
        agent_step.register_output_data_unit(agent_output_du)
        
        await agent_step.initialize()
        self.components['agent_step'] = agent_step
        
        # 5. Create triggers
        print("   Creating enhanced triggers...")
        
        # User input trigger
        user_trigger_config = TriggerConfig(
            trigger_type=TriggerType.DATA_UPDATED
        )
        user_trigger = DataUpdatedTrigger([user_input_du], user_trigger_config, name="enhanced_user_input_trigger")
        await user_trigger.add_callback(self._on_user_input)
        await user_trigger.start_monitoring()
        self.components['user_trigger'] = user_trigger
        
        # Agent input trigger
        agent_input_trigger_config = TriggerConfig(
            trigger_type=TriggerType.DATA_UPDATED
        )
        agent_input_trigger = DataUpdatedTrigger([agent_input_du], agent_input_trigger_config, name="enhanced_agent_input_trigger")
        await agent_input_trigger.add_callback(self._on_agent_input)
        await agent_input_trigger.start_monitoring()
        self.components['agent_input_trigger'] = agent_input_trigger
        
        # Agent output trigger
        agent_output_trigger_config = TriggerConfig(
            trigger_type=TriggerType.DATA_UPDATED
        )
        agent_output_trigger = DataUpdatedTrigger([agent_output_du], agent_output_trigger_config, name="enhanced_agent_output_trigger")
        await agent_output_trigger.add_callback(self._on_agent_output)
        await agent_output_trigger.start_monitoring()
        self.components['agent_output_trigger'] = agent_output_trigger
        
        # 6. Create direct links
        print("   Creating enhanced direct links...")
        
        # User to agent link
        user_to_agent_config = LinkConfig(
            link_type=LinkType.DIRECT
        )
        user_to_agent_link = DirectLink(user_input_du, agent_input_du, user_to_agent_config)
        await user_to_agent_link.start()  # Start the link
        self.components['user_to_agent_link'] = user_to_agent_link
        
        # 7. Create enhanced CLI interface
        print("   Creating enhanced CLI interface...")
        self.cli = EnhancedCLIInterface(
            user_input_du, agent_output_du, 
            self.history_manager, self.performance_tracker, agent_step
        )
        
        # Store CLI reference for output notifications
        self.components['cli'] = self.cli
        
        print("✅ Enhanced chat workflow setup complete!")
        print(f"📊 Performance tracking enabled")
        print(f"💾 Conversation history enabled")
        print(f"🤝 A2A agent collaboration enabled")
        print(f"🔧 MCP tool integration enabled")
        print(f"🚀 Ready for enhanced collaborative chat experience!")
        
    async def _on_user_input(self, data: Dict[str, Any]):
        """Handle user input trigger activation."""
        # Transfer data through the direct link
        user_to_agent_link = self.components.get('user_to_agent_link')
        if user_to_agent_link:
            await user_to_agent_link.transfer(data)
        
    async def _on_agent_input(self, data: Dict[str, Any]):
        """Handle agent input trigger activation."""
        # Process through agent step
        agent_step = self.components['agent_step']
        result = await agent_step.process(data)
        
        # Store result in output data unit
        agent_output_du = self.components['agent_output_du']
        await agent_output_du.set(result)
        
    async def _on_agent_output(self, data: Dict[str, Any]):
        """Handle agent output trigger activation."""
        # Notify CLI interface of new output
        cli = self.components.get('cli')
        if cli and hasattr(cli, '_on_output_received'):
            await cli._on_output_received(data)
    
    async def run(self):
        """Run the enhanced chat workflow."""
        print("\n🚀 Starting Enhanced Chat Experience...")
        print("=" * 60)
        
        await self.cli.start()
        
        # Keep running until CLI stops
        while self.cli.running:
            await asyncio.sleep(0.1)
    
    async def shutdown(self):
        """Shutdown the enhanced chat workflow."""
        print("\n🧹 Shutting down enhanced chat workflow...")
        
        if self.cli:
            await self.cli.stop()
        
        # Shutdown components in reverse order
        for name, component in reversed(list(self.components.items())):
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'stop_monitoring'):
                    await component.stop_monitoring()
                print(f"   ✅ Shutdown {name}")
            except Exception as e:
                print(f"   ⚠️  Error shutting down {name}: {e}")
        
        if self.executor:
            await self.executor.shutdown()
            print("   ✅ Shutdown executor")
        
        print("✅ Enhanced shutdown complete!")


async def main():
    """Main function to run the enhanced collaborative chat workflow demo."""
    workflow = EnhancedChatWorkflow()
    
    try:
        print("🚀 Starting Enhanced NanoBrain Collaborative Chat Workflow Demo")
        print("=" * 70)
        print("🤝 Features: A2A Agent Collaboration • MCP Tool Integration")
        print("📊 Enhanced: History • Metrics • Multi-turn Context • Export")
        print("=" * 70)
        
        await workflow.setup()
        await workflow.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await workflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())