# LinkDirect

Direct link implementation that transfers data directly.

Biological analogy: Fast, direct neural pathway.
Justification: Like how some neural pathways offer rapid, direct transmission
(e.g., reflexes), direct links provide immediate data transfer between components.

## Description

Direct link between data units with high reliability

## Biological Analogy

Myelinated axon connection

## Justification

Like how myelinated axons provide fast, reliable signal transmission between neurons, LinkDirect provides fast, reliable data transfer between components.


## Objectives

- Transfer data directly from source to target
- Maintain high reliability and minimal loss
- Support automatic data transfer on changes
- Adapt connection strength based on usage

## Default Configuration

```yaml
auto_transfer: true
connection_strength: 0.8
input_data: null
output_data: null
reliability: 0.95
signal_decay: 0.1
trigger: null
```

## Configuration Validation

### Required Parameters

- `input_data`
- `output_data`

### Optional Parameters

- `reliability`
- `connection_strength`
- `signal_decay`
- `trigger`
- `auto_transfer`

## Usage Examples

### Basic

Basic direct link with default settings

```yaml
auto_transfer: true
connection_strength: 0.8
reliability: 0.95
signal_decay: 0.1
```

### Manual

Manual direct link without auto transfer

```yaml
auto_transfer: false
connection_strength: 0.9
reliability: 0.99
signal_decay: 0.05
```

## Methods

### setup_trigger

```python
def setup_trigger(self, check_interval, debug)
```

Set up the trigger for this link.

Biological analogy: Establishing a signal detection mechanism.
Justification: Like how synapses establish signal detection mechanisms
to respond to neurotransmitters, this sets up a mechanism to detect
changes and trigger responses.

Args:
    check_interval: Interval (in seconds) for checking data changes
    debug: Whether to enable debug mode

### start_monitoring_sync

```python
def start_monitoring_sync(self)
```

Start monitoring for data changes synchronously (non-awaitable version).

This method is used when we need to start monitoring from a non-async context.
It creates a task for the async start_monitoring method.

