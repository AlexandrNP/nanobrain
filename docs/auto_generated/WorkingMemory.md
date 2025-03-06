# WorkingMemory

Short-term memory storage with limited capacity.

Biological analogy: Working memory in the prefrontal cortex.
Justification: Like how working memory has limited capacity and
follows recency-based retention, this class implements a capacity-limited
storage with LRU (Least Recently Used) eviction policy.

## Methods

### store

```python
def store(self, key: str, value: Any) -> bool
```

Store a value in working memory.

Biological analogy: Encoding information into working memory.
Justification: Like how the brain encodes new information into
working memory, potentially displacing older items when capacity
is reached.

### retrieve

```python
def retrieve(self, key: str) -> Optional[Any]
```

Retrieve a value from working memory.

Biological analogy: Retrieving information from working memory.
Justification: Like how retrieving information from working memory
makes it more likely to be retained (recency effect), this method
updates the item's recency status.

