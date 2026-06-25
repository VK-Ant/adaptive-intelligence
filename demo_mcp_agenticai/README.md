# adaptive-intelligence — MCP + Agentic AI Demo

## What's inside

```
demo_mcp_agenticai/
    README.md               # This file
    requirements.txt        # pip install -r requirements.txt
    demo_basic.py           # Basic usage + incremental learning
    demo_tools.py           # Tool registry + cost optimization
    demo_agentic.py         # Agentic multi-round retrieval
    demo_mcp_server.py      # Serve as MCP server
    demo_mcp_client.py      # Connect to MCP server
    tools/
        financial_tools.py  # Financial analysis tools
        healthcare_tools.py # Healthcare analysis tools
    data/
        financial/          # Financial documents
        healthcare/         # Healthcare documents
```

## Setup (2 minutes)

```bash
cd demo_mcp_agenticai
pip install -r requirements.txt
```

## Run demos

```bash
# 1. Basic usage + incremental learning
python demo_basic.py

# 2. Tools + cost optimization
python demo_tools.py

# 3. Agentic multi-round retrieval
python demo_agentic.py

# 4. MCP server (run in terminal 1)
python demo_mcp_server.py

# 5. MCP client (run in terminal 2, while server is running)
python demo_mcp_client.py
```

## No LLM needed

All demos work WITHOUT an LLM by default.
The system returns relevant document excerpts.
Add `--llm ollama` or `--llm groq` for synthesized answers.

## Scenarios

### Financial
- Q3 revenue analysis
- Supply chain risk assessment
- Cost optimization (RL learns to send fewer chunks)

### Healthcare
- Drug interaction lookup
- Treatment protocol queries
- Clinical trial data retrieval

## Author

Venkatkumar Rajan | @VK_Venkatkumar
https://github.com/VK-Ant/adaptive-intelligence
