"""
Conversation history management.

Persistent conversation storage and retrieval with search capabilities.
"""

import asyncio
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from nanobrain.core.data_unit import DataUnitBase


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    timestamp: datetime
    user_input: str
    agent_response: str
    response_time_ms: float
    conversation_id: str
    message_id: int
    metadata: Optional[Dict[str, Any]] = None


class ConversationHistoryUnit(DataUnitBase):
    """Persistent storage and retrieval of conversation data with search capabilities."""
    
    def __init__(self, database_adapter=None, table_name: str = "conversations", config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.database_adapter = database_adapter
        self.table_name = table_name
        self.db_path = self.config.get('db_path', 'conversation_history.db')
        self._db_lock = asyncio.Lock()
        
    async def _initialize_impl(self) -> None:
        """Initialize conversation history storage."""
        if self.database_adapter:
            await self.database_adapter.initialize()
        else:
            # Use SQLite directly if no adapter provided
            await self._init_sqlite_db()
            
    async def _init_sqlite_db(self) -> None:
        """Initialize SQLite database for conversation history."""
        async with self._db_lock:
            # Create database and table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_conversation_id ON {self.table_name}(conversation_id)')
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_timestamp ON {self.table_name}(timestamp)')
            
            conn.commit()
            conn.close()
            
    async def get(self) -> Any:
        """Get recent conversation data."""
        return await self.get_recent_conversations(hours=24)
        
    async def set(self, data: Any) -> None:
        """Set/save conversation message."""
        if isinstance(data, ConversationMessage):
            await self.save_message(data)
        elif isinstance(data, dict):
            message = ConversationMessage(**data)
            await self.save_message(message)
        else:
            raise TypeError("Data must be ConversationMessage or dict")
            
    async def clear(self) -> None:
        """Clear all conversation history."""
        async with self._db_lock:
            if self.database_adapter:
                await self.database_adapter.execute_query(f"DELETE FROM {self.table_name}")
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {self.table_name}")
                conn.commit()
                conn.close()
                
        self.logger.info(f"Cleared all conversation history from {self.table_name}")
        
    async def save_message(self, message: ConversationMessage) -> None:
        """Save a conversation message."""
        async with self._db_lock:
            if self.database_adapter:
                await self.database_adapter.execute_query(
                    f"""INSERT INTO {self.table_name} 
                       (conversation_id, message_id, timestamp, user_input, agent_response, response_time_ms, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    {
                        'conversation_id': message.conversation_id,
                        'message_id': message.message_id,
                        'timestamp': message.timestamp.isoformat(),
                        'user_input': message.user_input,
                        'agent_response': message.agent_response,
                        'response_time_ms': message.response_time_ms,
                        'metadata': str(message.metadata) if message.metadata else None
                    }
                )
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"""
                    INSERT INTO {self.table_name} 
                    (conversation_id, message_id, timestamp, user_input, agent_response, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.conversation_id,
                    message.message_id,
                    message.timestamp.isoformat(),
                    message.user_input,
                    message.agent_response,
                    message.response_time_ms,
                    str(message.metadata) if message.metadata else None
                ))
                conn.commit()
                conn.close()
                
        self.logger.debug(f"Saved message for conversation {message.conversation_id}")
        
    async def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Get conversation history for a specific conversation."""
        async with self._db_lock:
            if self.database_adapter:
                result = await self.database_adapter.execute_query(
                    f"""SELECT * FROM {self.table_name} 
                       WHERE conversation_id = ? 
                       ORDER BY timestamp DESC LIMIT ?""",
                    {'conversation_id': conversation_id, 'limit': limit}
                )
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_name} 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (conversation_id, limit))
                result = cursor.fetchall()
                conn.close()
                
        messages = []
        for row in result:
            messages.append(ConversationMessage(
                timestamp=datetime.fromisoformat(row[3]),
                user_input=row[4],
                agent_response=row[5],
                response_time_ms=row[6],
                conversation_id=row[1],
                message_id=row[2],
                metadata=eval(row[7]) if row[7] else None
            ))
            
        return messages
        
    async def get_recent_conversations(self, hours: int = 24) -> List[str]:
        """Get list of recent conversation IDs."""
        async with self._db_lock:
            cutoff_time = datetime.now().replace(hour=datetime.now().hour - hours)
            
            if self.database_adapter:
                result = await self.database_adapter.execute_query(
                    f"""SELECT DISTINCT conversation_id FROM {self.table_name} 
                       WHERE timestamp > ? 
                       ORDER BY timestamp DESC""",
                    {'cutoff_time': cutoff_time.isoformat()}
                )
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT DISTINCT conversation_id FROM {self.table_name} 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                result = cursor.fetchall()
                conn.close()
                
        return [row[0] for row in result]
        
    async def search_conversations(self, query: str, limit: int = 100) -> List[ConversationMessage]:
        """Search conversations by content."""
        async with self._db_lock:
            search_query = f"%{query}%"
            
            if self.database_adapter:
                result = await self.database_adapter.execute_query(
                    f"""SELECT * FROM {self.table_name} 
                       WHERE user_input LIKE ? OR agent_response LIKE ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    {'query1': search_query, 'query2': search_query, 'limit': limit}
                )
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {self.table_name} 
                    WHERE user_input LIKE ? OR agent_response LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (search_query, search_query, limit))
                result = cursor.fetchall()
                conn.close()
                
        messages = []
        for row in result:
            messages.append(ConversationMessage(
                timestamp=datetime.fromisoformat(row[3]),
                user_input=row[4],
                agent_response=row[5],
                response_time_ms=row[6],
                conversation_id=row[1],
                message_id=row[2],
                metadata=eval(row[7]) if row[7] else None
            ))
            
        return messages
        
    async def export_conversations(self, output_file: str, conversation_ids: Optional[List[str]] = None) -> None:
        """Export conversations to file."""
        import json
        
        if conversation_ids:
            all_messages = []
            for conv_id in conversation_ids:
                messages = await self.get_conversation_history(conv_id, limit=1000)
                all_messages.extend(messages)
        else:
            # Export all conversations
            async with self._db_lock:
                if self.database_adapter:
                    result = await self.database_adapter.execute_query(f"SELECT * FROM {self.table_name} ORDER BY timestamp")
                else:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM {self.table_name} ORDER BY timestamp")
                    result = cursor.fetchall()
                    conn.close()
                    
            all_messages = []
            for row in result:
                all_messages.append(asdict(ConversationMessage(
                    timestamp=datetime.fromisoformat(row[3]),
                    user_input=row[4],
                    agent_response=row[5],
                    response_time_ms=row[6],
                    conversation_id=row[1],
                    message_id=row[2],
                    metadata=eval(row[7]) if row[7] else None
                )))
                
        # Convert datetime objects to strings for JSON serialization
        for message in all_messages:
            if isinstance(message.get('timestamp'), datetime):
                message['timestamp'] = message['timestamp'].isoformat()
                
        with open(output_file, 'w') as f:
            json.dump(all_messages, f, indent=2, default=str)
            
        self.logger.info(f"Exported {len(all_messages)} messages to {output_file}")
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        async with self._db_lock:
            if self.database_adapter:
                result = await self.database_adapter.execute_query(f"""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(DISTINCT conversation_id) as total_conversations,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(timestamp) as earliest_message,
                        MAX(timestamp) as latest_message
                    FROM {self.table_name}
                """)
            else:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(DISTINCT conversation_id) as total_conversations,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(timestamp) as earliest_message,
                        MAX(timestamp) as latest_message
                    FROM {self.table_name}
                """)
                result = cursor.fetchone()
                conn.close()
                
        if result:
            return {
                'total_messages': result[0],
                'total_conversations': result[1],
                'avg_response_time_ms': result[2],
                'earliest_message': result[3],
                'latest_message': result[4]
            }
        return {} 