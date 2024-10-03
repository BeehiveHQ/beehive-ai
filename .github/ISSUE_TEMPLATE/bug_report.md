---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Beehive structure**
Create an example Beehive that replicates your Beehive structure:
```python
agent1 = BeehiveAgent(...)
agent2 = BeehiveAgent(...)
beehive = Beehive(
  name=...
  backstory=...,
  execution_process=FixedExecution(route=(agent1 >> agent2))
)
```

**Output & Traceback**
Paste a concise version of the output / traceback below:
```
stdout
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Desktop (please complete the following information):**
 - OS: [e.g., iOS]
 - Python version: [e.g., 3.7]

**Additional context**
Add any other context about the problem here.
