"""
Session lifecycle management.

Session and context management for user interactions.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from nanobrain.core.data_unit import DataUnitBase


@dataclass
class SessionData:
    """Represents session data."""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class SessionManager(DataUnitBase):
    """Session lifecycle and metadata handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour default
        self.max_sessions = self.config.get('max_sessions', 1000)
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        self._sessions: Dict[str, SessionData] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def _initialize_impl(self) -> None:
        """Initialize session manager."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        self.logger.info(f"Session manager initialized with timeout {self.session_timeout}s")
        
    async def _shutdown_impl(self) -> None:
        """Shutdown session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Session manager shutdown")
        
    async def get(self) -> Any:
        """Get all active sessions."""
        async with self._lock:
            return {sid: asdict(session) for sid, session in self._sessions.items()}
            
    async def set(self, data: Any) -> None:
        """Set session data."""
        if isinstance(data, dict) and 'session_id' in data:
            await self.create_session(
                data['session_id'],
                data.get('user_id', 'anonymous'),
                data.get('data', {})
            )
        else:
            raise TypeError("Data must be dict with session_id")
            
    async def clear(self) -> None:
        """Clear all sessions."""
        async with self._lock:
            self._sessions.clear()
            self.logger.info("All sessions cleared")
            
    async def create_session(self, session_id: str, user_id: str, initial_data: Optional[Dict[str, Any]] = None) -> SessionData:
        """Create a new session."""
        async with self._lock:
            now = datetime.now()
            session = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_accessed=now,
                data=initial_data or {},
                metadata={'created_by': 'session_manager'}
            )
            
            # Check session limit
            if len(self._sessions) >= self.max_sessions:
                await self._cleanup_oldest_sessions(1)
                
            self._sessions[session_id] = session
            self.logger.debug(f"Created session {session_id} for user {user_id}")
            return session
            
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Update last accessed time
                session.last_accessed = datetime.now()
                return session
            return None
            
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.data.update(data)
                session.last_accessed = datetime.now()
                self.logger.debug(f"Updated session {session_id}")
                return True
            return False
            
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self.logger.debug(f"Deleted session {session_id}")
                return True
            return False
            
    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for a user."""
        async with self._lock:
            return [session for session in self._sessions.values() if session.user_id == user_id]
            
    async def is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and not expired."""
        session = await self.get_session(session_id)
        if not session:
            return False
            
        # Check if session has expired
        expiry_time = session.last_accessed + timedelta(seconds=self.session_timeout)
        return datetime.now() < expiry_time
        
    async def extend_session(self, session_id: str) -> bool:
        """Extend session timeout."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_accessed = datetime.now()
                self.logger.debug(f"Extended session {session_id}")
                return True
            return False
            
    async def get_session_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get specific data from session."""
        session = await self.get_session(session_id)
        if session:
            return session.data.get(key, default)
        return default
        
    async def set_session_data(self, session_id: str, key: str, value: Any) -> bool:
        """Set specific data in session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.data[key] = value
                session.last_accessed = datetime.now()
                return True
            return False
            
    async def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        async with self._lock:
            now = datetime.now()
            active_count = 0
            for session in self._sessions.values():
                expiry_time = session.last_accessed + timedelta(seconds=self.session_timeout)
                if now < expiry_time:
                    active_count += 1
            return active_count
            
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        async with self._lock:
            now = datetime.now()
            total_sessions = len(self._sessions)
            active_sessions = 0
            expired_sessions = 0
            
            user_counts = {}
            oldest_session = None
            newest_session = None
            
            for session in self._sessions.values():
                expiry_time = session.last_accessed + timedelta(seconds=self.session_timeout)
                if now < expiry_time:
                    active_sessions += 1
                else:
                    expired_sessions += 1
                    
                # Count sessions per user
                user_counts[session.user_id] = user_counts.get(session.user_id, 0) + 1
                
                # Track oldest and newest sessions
                if oldest_session is None or session.created_at < oldest_session:
                    oldest_session = session.created_at
                if newest_session is None or session.created_at > newest_session:
                    newest_session = session.created_at
                    
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'expired_sessions': expired_sessions,
                'unique_users': len(user_counts),
                'sessions_per_user': user_counts,
                'oldest_session': oldest_session.isoformat() if oldest_session else None,
                'newest_session': newest_session.isoformat() if newest_session else None,
                'session_timeout': self.session_timeout,
                'max_sessions': self.max_sessions
            }
            
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._remove_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")
                
    async def _remove_expired_sessions(self) -> int:
        """Remove expired sessions."""
        async with self._lock:
            now = datetime.now()
            expired_sessions = []
            
            for session_id, session in self._sessions.items():
                expiry_time = session.last_accessed + timedelta(seconds=self.session_timeout)
                if now >= expiry_time:
                    expired_sessions.append(session_id)
                    
            for session_id in expired_sessions:
                del self._sessions[session_id]
                
            if expired_sessions:
                self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
                
            return len(expired_sessions)
            
    async def _cleanup_oldest_sessions(self, count: int) -> None:
        """Remove oldest sessions to make room."""
        if not self._sessions:
            return
            
        # Sort sessions by creation time
        sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].created_at)
        
        for i in range(min(count, len(sorted_sessions))):
            session_id = sorted_sessions[i][0]
            del self._sessions[session_id]
            self.logger.debug(f"Removed oldest session {session_id}")
            
    async def export_sessions(self, output_file: str) -> None:
        """Export sessions to file."""
        async with self._lock:
            sessions_data = []
            for session in self._sessions.values():
                session_dict = asdict(session)
                # Convert datetime objects to strings
                session_dict['created_at'] = session.created_at.isoformat()
                session_dict['last_accessed'] = session.last_accessed.isoformat()
                sessions_data.append(session_dict)
                
        with open(output_file, 'w') as f:
            json.dump(sessions_data, f, indent=2, default=str)
            
        self.logger.info(f"Exported {len(sessions_data)} sessions to {output_file}")
        
    async def import_sessions(self, input_file: str) -> int:
        """Import sessions from file."""
        with open(input_file, 'r') as f:
            sessions_data = json.load(f)
            
        imported_count = 0
        async with self._lock:
            for session_dict in sessions_data:
                session = SessionData(
                    session_id=session_dict['session_id'],
                    user_id=session_dict['user_id'],
                    created_at=datetime.fromisoformat(session_dict['created_at']),
                    last_accessed=datetime.fromisoformat(session_dict['last_accessed']),
                    data=session_dict['data'],
                    metadata=session_dict.get('metadata')
                )
                self._sessions[session.session_id] = session
                imported_count += 1
                
        self.logger.info(f"Imported {imported_count} sessions from {input_file}")
        return imported_count 