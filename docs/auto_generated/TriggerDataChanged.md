# TriggerDataChanged

Trigger that activates when data has changed.

Biological analogy: Change detection neurons.
Justification: Similar to how some neurons in the visual system respond specifically
to changes in the visual field, this trigger responds specifically to changes in data.

## Description

Trigger that activates when input data changes

## Biological Analogy

Change detection neurons

## Justification

Like how certain neurons in the visual system respond specifically to changes in the environment, this trigger responds to changes in data.


## Objectives

- Detect changes in input data
- Activate runnable when data changes
- Control sensitivity to changes
- Manage monitoring of data sources

## Default Configuration

```yaml
activation_threshold: 0.5
monitoring_interval: 0.1
runnable: null
sensitivity: 0.8
```

## Configuration Validation

### Required Parameters

- `runnable`

### Optional Parameters

- `sensitivity`
- `activation_threshold`
- `monitoring_interval`

## Usage Examples

### Basic

Basic data change trigger

```yaml
activation_threshold: 0.5
monitoring_interval: 0.1
sensitivity: 0.8
```

## Methods

### Constructor

```python
def __init__(self, source_step, runnable, **kwargs)
```

Initialize a data change trigger.

Args:
    source_step: The step whose output is monitored for changes
    runnable: The entity to run when triggered (typically a link)

### start_monitoring

```python
def start_monitoring(self)
```

Start continuous monitoring for data changes.

Biological analogy: Activating attention system.
Justification: Like how the brain's attention system activates to monitor
for specific stimuli, this method activates continuous monitoring for data changes.

### stop_monitoring

```python
def stop_monitoring(self)
```

Stop continuous monitoring for data changes.

Biological analogy: Deactivating attention system.
Justification: Like how the brain's attention system deactivates when no longer
needed, this method deactivates continuous monitoring for data changes.

### runnable

```python
def runnable(self, value)
```

Set the runnable object.

This also resets the last_data to ensure proper change detection.

### sensitivity

```python
def sensitivity(self, value)
```

Set the sensitivity level.

### adaptation_rate

```python
def adaptation_rate(self, value)
```

Set the adaptation rate.

### activation_gate

```python
def activation_gate(self, value)
```

Set the activation gate.

