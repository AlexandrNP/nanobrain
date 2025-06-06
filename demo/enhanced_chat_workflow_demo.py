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

Architecture:
CLI Input → User Input DataUnit → Agent Input DataUnit → ConversationalAgentStep → Agent Output DataUnit → CLI Output
                                                    ↓
                                            Conversation History
                                            Performance Metrics
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

from core.data_unit import DataUnitMemory, DataUnitConfig
from core.trigger import DataUpdatedTrigger, TriggerConfig
from core.link import DirectLink, LinkConfig
from core.step import Step, StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import LocalExecutor, ExecutorConfig
from config.component_factory import ComponentFactory, get_factory


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
    Enhanced step wrapper for ConversationalAgent with metrics and history.
    """
    
    def __init__(self, config: StepConfig, agent: ConversationalAgent, 
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
        
    async def start(self):
        """Start the enhanced CLI interface."""
        self.running = True
        
        # Set up output monitoring
        await self.output_data_unit.subscribe(self._on_output_received)
        
        self._show_welcome()
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        # Start metrics display thread if enabled
        if self.show_metrics:
            self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self.metrics_thread.start()
        
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
        print("🚀 Features: History • Metrics • Multi-turn Context • Export")
        print("📝 Type your messages below or use commands:")
        print("   /help     - Show all commands")
        print("   /quit     - Exit the chat")
        print("   /new      - Start new conversation")
        print("   /history  - Show conversation history")
        print("   /stats    - Show performance statistics")
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
                    asyncio.run_coroutine_threadsafe(
                        self._handle_command(user_input),
                        asyncio.get_event_loop()
                    )
                    continue
                
                # Handle regular chat input
                asyncio.run_coroutine_threadsafe(
                    self.input_data_unit.store({'user_input': user_input}),
                    asyncio.get_event_loop()
                )
                
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Input error: {e}")
    
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
    
    async def _on_output_received(self, data: Dict[str, Any]):
        """Handle output from the agent."""
        response = data.get('agent_response', '')
        if response and response.strip():
            if self.show_metrics:
                print()  # New line after metrics
            print(f"\n🤖 Assistant: {response}")
    
    def _show_help(self):
        """Show comprehensive help information."""
        print("\n📋 Enhanced Chat Commands:")
        print("  /help      - Show this help message")
        print("  /quit      - Exit the chat")
        print("  /new       - Start a new conversation")
        print("  /history   - Show current conversation history")
        print("  /stats     - Show performance statistics")
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


class EnhancedChatWorkflow:
    """
    Enhanced chat workflow orchestrator with additional features.
    """
    
    def __init__(self):
        self.factory = get_factory()
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
            data_type="memory",
            name="user_input",
            description="Enhanced user input with metadata",
            persistent=False,
            cache_size=200  # Increased cache
        )
        user_input_du = DataUnitMemory(user_input_config)
        await user_input_du.initialize()
        self.components['user_input_du'] = user_input_du
        
        # Agent input data unit
        agent_input_config = DataUnitConfig(
            data_type="memory",
            name="agent_input",
            description="Enhanced agent input with context",
            persistent=False,
            cache_size=200
        )
        agent_input_du = DataUnitMemory(agent_input_config)
        await agent_input_du.initialize()
        self.components['agent_input_du'] = agent_input_du
        
        # Agent output data unit
        agent_output_config = DataUnitConfig(
            data_type="memory",
            name="agent_output",
            description="Enhanced agent output with metrics",
            persistent=False,
            cache_size=200
        )
        agent_output_du = DataUnitMemory(agent_output_config)
        await agent_output_du.initialize()
        self.components['agent_output_du'] = agent_output_du
        
        # 3. Create conversational agent
        print("   Creating enhanced conversational agent...")
        agent_config = AgentConfig(
            name="EnhancedChatAssistant",
            description="Enhanced conversational assistant with context management",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500,  # Increased for longer responses
            system_prompt="""You are an advanced AI assistant with enhanced capabilities. You maintain conversation context, provide detailed and helpful responses, and can engage in complex multi-turn conversations.

Key capabilities:
- Maintain conversation context across multiple turns
- Provide detailed explanations with examples
- Ask clarifying questions when needed
- Offer suggestions and follow-up topics
- Adapt your communication style to the user's needs
- Remember previous topics in the conversation
- Provide structured responses when appropriate

Guidelines:
- Be conversational, helpful, and engaging
- Show enthusiasm for learning and helping
- Provide accurate and well-reasoned responses
- Use examples and analogies when helpful
- Acknowledge when you don't know something
- Offer to explore topics in more depth""",
            auto_initialize=False,
            debug_mode=True,
            enable_logging=True,
            log_conversations=True,
            log_tool_calls=True
        )
        
        agent = ConversationalAgent(agent_config)
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
            trigger_type="data_updated",
            name="enhanced_user_input_trigger",
            description="Enhanced trigger for user input processing"
        )
        user_trigger = DataUpdatedTrigger([user_input_du], user_trigger_config, name="enhanced_user_input_trigger")
        await user_trigger.add_callback(self._on_user_input)
        await user_trigger.start_monitoring()
        self.components['user_trigger'] = user_trigger
        
        # Agent input trigger
        agent_input_trigger_config = TriggerConfig(
            trigger_type="data_updated",
            name="enhanced_agent_input_trigger",
            description="Enhanced trigger for agent input processing"
        )
        agent_input_trigger = DataUpdatedTrigger([agent_input_du], agent_input_trigger_config, name="enhanced_agent_input_trigger")
        await agent_input_trigger.add_callback(self._on_agent_input)
        await agent_input_trigger.start_monitoring()
        self.components['agent_input_trigger'] = agent_input_trigger
        
        # Agent output trigger
        agent_output_trigger_config = TriggerConfig(
            trigger_type="data_updated",
            name="enhanced_agent_output_trigger",
            description="Enhanced trigger for agent output processing"
        )
        agent_output_trigger = DataUpdatedTrigger([agent_output_du], agent_output_trigger_config, name="enhanced_agent_output_trigger")
        await agent_output_trigger.add_callback(self._on_agent_output)
        await agent_output_trigger.start_monitoring()
        self.components['agent_output_trigger'] = agent_output_trigger
        
        # 6. Create direct links
        print("   Creating enhanced direct links...")
        
        # User to agent link
        user_to_agent_config = LinkConfig(
            link_type="direct",
            name="enhanced_user_to_agent_link",
            description="Enhanced link from user input to agent input"
        )
        user_to_agent_link = DirectLink(user_input_du, agent_input_du, user_to_agent_config)
        self.components['user_to_agent_link'] = user_to_agent_link
        
        # 7. Create enhanced CLI interface
        print("   Creating enhanced CLI interface...")
        self.cli = EnhancedCLIInterface(
            user_input_du, agent_output_du, 
            self.history_manager, self.performance_tracker, agent_step
        )
        
        print("✅ Enhanced chat workflow setup complete!")
        print(f"📊 Performance tracking enabled")
        print(f"💾 Conversation history enabled")
        print(f"🚀 Ready for enhanced chat experience!")
        
    async def _on_user_input(self, data: Dict[str, Any]):
        """Handle user input trigger activation."""
        # Data is automatically transferred by the direct link
        pass
        
    async def _on_agent_input(self, data: Dict[str, Any]):
        """Handle agent input trigger activation."""
        # Process through agent step
        agent_step = self.components['agent_step']
        result = await agent_step.process(data)
        
        # Store result in output data unit
        agent_output_du = self.components['agent_output_du']
        await agent_output_du.store(result)
        
    async def _on_agent_output(self, data: Dict[str, Any]):
        """Handle agent output trigger activation."""
        # Output is handled by CLI interface subscription
        pass
    
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
                    component.stop_monitoring()
                print(f"   ✅ Shutdown {name}")
            except Exception as e:
                print(f"   ⚠️  Error shutting down {name}: {e}")
        
        if self.executor:
            await self.executor.shutdown()
            print("   ✅ Shutdown executor")
        
        print("✅ Enhanced shutdown complete!")


async def main():
    """Main function to run the enhanced chat workflow demo."""
    workflow = EnhancedChatWorkflow()
    
    try:
        print("🚀 Starting Enhanced NanoBrain Chat Workflow Demo")
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