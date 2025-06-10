"""
Delegation engine implementation.

Rule-based task delegation and routing for collaborative agents.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from nanobrain.core.logging_system import get_logger


@dataclass
class DelegationRule:
    """Represents a delegation rule."""
    rule_id: str
    keywords: List[str]
    target: str
    target_type: str  # 'a2a_agent', 'mcp_tool', 'custom'
    description: str
    priority: int = 1
    confidence_threshold: float = 0.8
    conditions: Optional[Dict[str, Any]] = None
    enabled: bool = True


class DelegationEngine:
    """Rule-based task delegation and routing."""
    
    def __init__(self, initial_rules: Optional[List[Dict[str, Any]]] = None):
        self.rules: List[DelegationRule] = []
        self._lock = asyncio.Lock()
        self.logger = get_logger("delegation_engine")
        
        # Load initial rules
        if initial_rules:
            for rule_data in initial_rules:
                asyncio.create_task(self.add_rule(rule_data))
                
    async def add_rule(self, rule_data: Dict[str, Any]) -> str:
        """Add a new delegation rule."""
        async with self._lock:
            rule_id = rule_data.get('rule_id', f"rule_{len(self.rules) + 1}")
            
            rule = DelegationRule(
                rule_id=rule_id,
                keywords=rule_data.get('keywords', []),
                target=rule_data.get('target', ''),
                target_type=rule_data.get('target_type', 'custom'),
                description=rule_data.get('description', ''),
                priority=rule_data.get('priority', 1),
                confidence_threshold=rule_data.get('confidence_threshold', 0.8),
                conditions=rule_data.get('conditions'),
                enabled=rule_data.get('enabled', True)
            )
            
            self.rules.append(rule)
            # Sort rules by priority (higher priority first)
            self.rules.sort(key=lambda r: r.priority, reverse=True)
            
            self.logger.debug(f"Added delegation rule: {rule_id}")
            return rule_id
            
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a delegation rule."""
        async with self._lock:
            for i, rule in enumerate(self.rules):
                if rule.rule_id == rule_id:
                    del self.rules[i]
                    self.logger.debug(f"Removed delegation rule: {rule_id}")
                    return True
            return False
            
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing delegation rule."""
        async with self._lock:
            for rule in self.rules:
                if rule.rule_id == rule_id:
                    # Update rule attributes
                    for key, value in updates.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)
                    
                    # Re-sort if priority changed
                    if 'priority' in updates:
                        self.rules.sort(key=lambda r: r.priority, reverse=True)
                        
                    self.logger.debug(f"Updated delegation rule: {rule_id}")
                    return True
            return False
            
    async def enable_rule(self, rule_id: str) -> bool:
        """Enable a delegation rule."""
        return await self.update_rule(rule_id, {'enabled': True})
        
    async def disable_rule(self, rule_id: str) -> bool:
        """Disable a delegation rule."""
        return await self.update_rule(rule_id, {'enabled': False})
        
    async def should_delegate(self, input_text: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Check if input should be delegated based on rules."""
        async with self._lock:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                    
                confidence = await self._calculate_confidence(rule, input_text, **kwargs)
                
                if confidence >= rule.confidence_threshold:
                    # Check additional conditions if specified
                    if rule.conditions and not await self._check_conditions(rule.conditions, input_text, **kwargs):
                        continue
                        
                    self.logger.info(f"Delegation rule matched: {rule.rule_id} (confidence: {confidence:.2f})")
                    
                    return {
                        'rule_id': rule.rule_id,
                        'target': rule.target,
                        'type': rule.target_type,
                        'confidence': confidence,
                        'description': rule.description
                    }
                    
        return None
        
    async def _calculate_confidence(self, rule: DelegationRule, input_text: str, **kwargs) -> float:
        """Calculate confidence score for a rule match."""
        if not rule.keywords:
            return 0.0
            
        input_lower = input_text.lower()
        matched_keywords = 0
        
        for keyword in rule.keywords:
            # Support both exact matches and regex patterns
            if keyword.startswith('regex:'):
                pattern = keyword[6:]  # Remove 'regex:' prefix
                if re.search(pattern, input_lower, re.IGNORECASE):
                    matched_keywords += 1
            else:
                if keyword.lower() in input_lower:
                    matched_keywords += 1
                    
        # Calculate confidence as percentage of matched keywords
        confidence = matched_keywords / len(rule.keywords)
        
        # Apply priority boost
        priority_boost = (rule.priority - 1) * 0.1
        confidence = min(1.0, confidence + priority_boost)
        
        return confidence
        
    async def _check_conditions(self, conditions: Dict[str, Any], input_text: str, **kwargs) -> bool:
        """Check additional conditions for rule matching."""
        for condition_type, condition_value in conditions.items():
            if condition_type == 'min_length':
                if len(input_text) < condition_value:
                    return False
            elif condition_type == 'max_length':
                if len(input_text) > condition_value:
                    return False
            elif condition_type == 'required_context':
                # Check if required context keys are present in kwargs
                for key in condition_value:
                    if key not in kwargs:
                        return False
            elif condition_type == 'excluded_keywords':
                # Check that none of the excluded keywords are present
                input_lower = input_text.lower()
                for excluded_keyword in condition_value:
                    if excluded_keyword.lower() in input_lower:
                        return False
            elif condition_type == 'user_role':
                # Check user role if provided in kwargs
                user_role = kwargs.get('user_role', 'user')
                if user_role not in condition_value:
                    return False
                    
        return True
        
    async def get_rules(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Get all delegation rules."""
        async with self._lock:
            rules_data = []
            for rule in self.rules:
                if enabled_only and not rule.enabled:
                    continue
                    
                rules_data.append({
                    'rule_id': rule.rule_id,
                    'keywords': rule.keywords,
                    'target': rule.target,
                    'target_type': rule.target_type,
                    'description': rule.description,
                    'priority': rule.priority,
                    'confidence_threshold': rule.confidence_threshold,
                    'conditions': rule.conditions,
                    'enabled': rule.enabled
                })
                
            return rules_data
            
    async def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific delegation rule."""
        async with self._lock:
            for rule in self.rules:
                if rule.rule_id == rule_id:
                    return {
                        'rule_id': rule.rule_id,
                        'keywords': rule.keywords,
                        'target': rule.target,
                        'target_type': rule.target_type,
                        'description': rule.description,
                        'priority': rule.priority,
                        'confidence_threshold': rule.confidence_threshold,
                        'conditions': rule.conditions,
                        'enabled': rule.enabled
                    }
            return None
            
    async def test_rule(self, rule_id: str, test_input: str, **kwargs) -> Dict[str, Any]:
        """Test a rule against input without triggering delegation."""
        rule_data = await self.get_rule(rule_id)
        if not rule_data:
            return {'error': 'Rule not found'}
            
        # Create temporary rule object
        rule = DelegationRule(**rule_data)
        confidence = await self._calculate_confidence(rule, test_input, **kwargs)
        
        would_match = confidence >= rule.confidence_threshold
        conditions_met = True
        
        if rule.conditions:
            conditions_met = await self._check_conditions(rule.conditions, test_input, **kwargs)
            
        return {
            'rule_id': rule_id,
            'confidence': confidence,
            'threshold': rule.confidence_threshold,
            'would_match': would_match and conditions_met,
            'conditions_met': conditions_met,
            'test_input': test_input
        }
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get delegation engine statistics."""
        async with self._lock:
            enabled_rules = sum(1 for rule in self.rules if rule.enabled)
            disabled_rules = len(self.rules) - enabled_rules
            
            target_types = {}
            priorities = {}
            
            for rule in self.rules:
                target_types[rule.target_type] = target_types.get(rule.target_type, 0) + 1
                priorities[rule.priority] = priorities.get(rule.priority, 0) + 1
                
            return {
                'total_rules': len(self.rules),
                'enabled_rules': enabled_rules,
                'disabled_rules': disabled_rules,
                'target_types': target_types,
                'priority_distribution': priorities
            }
            
    async def export_rules(self, output_file: str):
        """Export rules to file."""
        import json
        
        rules_data = await self.get_rules()
        
        export_data = {
            'export_timestamp': asyncio.get_event_loop().time(),
            'total_rules': len(rules_data),
            'rules': rules_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Exported {len(rules_data)} rules to {output_file}")
        
    async def import_rules(self, input_file: str, replace_existing: bool = False):
        """Import rules from file."""
        import json
        
        with open(input_file, 'r') as f:
            import_data = json.load(f)
            
        if replace_existing:
            async with self._lock:
                self.rules.clear()
                
        imported_count = 0
        for rule_data in import_data.get('rules', []):
            await self.add_rule(rule_data)
            imported_count += 1
            
        self.logger.info(f"Imported {imported_count} rules from {input_file}")
        return imported_count
        
    async def clear_rules(self):
        """Clear all delegation rules."""
        async with self._lock:
            rule_count = len(self.rules)
            self.rules.clear()
            self.logger.info(f"Cleared {rule_count} delegation rules") 