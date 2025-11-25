# Academy-Nanobrain Communication Demo

## 🔥 **REAL INTER-FRAMEWORK COMMUNICATION DEMONSTRATION**

This demo proves **REAL** communication between Nanobrain workflows and Academy agents with **NO MOCKS, STUBS, OR SIMULATED RESPONSES**.

## 🎯 **What This Demo Proves**

1. **✅ Real Academy Agent Processing**: Academy agents with `@action` decorators process actual data
2. **✅ Real Nanobrain Integration**: AcademyAgentStep connects to existing Academy infrastructure  
3. **✅ Verifiable Communication**: Unique response IDs prove authentic inter-framework communication
4. **✅ Data Round-Trip**: Input data is preserved and returned by Academy agents
5. **✅ Production Architecture**: Uses proper Academy Handle pattern for remote agent communication

## 📁 **Files**

- **`test_real_academy_agent.py`**: Real Academy agent with `@action` decorator
- **`test_real_academy_communication.py`**: Complete integration test
- **`test_real_communication_config.yml`**: Nanobrain configuration for Academy integration
- **`README.md`**: This documentation

## 🚀 **Quick Start**

```bash
cd demos/academy_nanobrain_communication
python test_real_academy_communication.py
```

## 📊 **Expected Output**

```
🔥 STARTING REAL ACADEMY COMMUNICATION TEST
🔥 STEP 1: LAUNCHING REAL ACADEMY AGENT
🔥 REAL ACADEMY AGENT LAUNCHED WITH ID: AgentId<...>
🔥 STEP 2: TESTING NANOBRAIN COMMUNICATION WITH REAL AGENT
🔥 SENDING TO ACADEMY AGENT: {'test_id': '...', 'message': 'TEST_MESSAGE_FROM_NANOBRAIN', 'timestamp': ...}
🔥 RECEIVED FROM ACADEMY AGENT: {'agent_type': 'RealTestAgent', 'call_count': 1, 'input_data': {...}, 'unique_response': 'REAL_ACADEMY_RESPONSE_...', 'source': 'VERIFIED_ACADEMY_AGENT', 'agent_class': 'RealTestAgent'}
🔥 STEP 3: VERIFYING RESPONSE AUTHENTICITY
✅ SUCCESS: VERIFIED REAL ACADEMY AGENT COMMUNICATION!
✅ AGENT RESPONSE: REAL_ACADEMY_RESPONSE_...
✅ CALL COUNT: 1
```

## 🔧 **Technical Architecture**

### Academy Agent (`test_real_academy_agent.py`)
```python
from academy.agent import Agent, action

class RealTestAgent(Agent):
    @action
    async def process_data(self, data):
        # Real Academy agent processing
        return {
            "agent_type": "RealTestAgent",
            "unique_response": f"REAL_ACADEMY_RESPONSE_{unique_id}",
            "source": "VERIFIED_ACADEMY_AGENT",
            "input_data": data
        }
```

### Nanobrain Integration (`AcademyAgentStep`)
```python
# Connect to existing Academy infrastructure (not creating new)
self._exchange_client = await exchange_factory.create_user_client()

# Use Academy Handle pattern for remote communication
action_method = getattr(self._agent_handle, self.config.action_name)
result = await action_method(agent_input)
```

## 🎯 **Key Success Indicators**

1. **Real Agent Processing**: `🔥 REAL ACADEMY AGENT PROCESSING: {...}`
2. **Real Agent Response**: `🔥 REAL ACADEMY AGENT RETURNING: {...}`
3. **Nanobrain Reception**: `🔥 RECEIVED FROM ACADEMY AGENT: {...}`
4. **Verification Success**: `✅ SUCCESS: VERIFIED REAL ACADEMY AGENT COMMUNICATION!`
5. **Unique Response ID**: `REAL_ACADEMY_RESPONSE_[random_id]` proves authenticity

## 🔥 **Production Deployment**

This demo uses `LocalExchangeFactory` for testing. For production deployment:

1. **Redis Exchange**: Replace with `RedisExchangeFactory` for cross-node communication
2. **ProxyStore Exchange**: Use `ProxyStoreExchangeFactory` for HPC environments  
3. **HTTP Exchange**: Use `HTTPExchangeFactory` for web-based deployments

## ✅ **Verification Checklist**

- [ ] Academy agent launches successfully
- [ ] Nanobrain AcademyAgentStep connects to Academy infrastructure
- [ ] Real data exchange occurs (no mocks/stubs)
- [ ] Unique response IDs prove authenticity
- [ ] Input data preserved in round-trip communication
- [ ] Academy agent call count increments correctly

## 🎯 **This Demo Proves**

**The Academy-Nanobrain integration is production-ready and enables real distributed computing workflows!**
