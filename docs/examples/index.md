---
title: ZIO LangChain Examples
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# ZIO LangChain Examples

This section contains documentation for the example applications provided with ZIO LangChain. These examples demonstrate various capabilities of the library and serve as starting points for your own applications.

## Available Examples

- [Simple Chat](simple-chat.md) - Basic chat application demonstrating LLM interaction
- [Simple RAG](simple-rag.md) - Retrieval-Augmented Generation with document retrieval
- [Advanced Chat](advanced-chat.md) - Chat with streaming responses and memory
- [Enhanced RAG](enhanced-rag.md) - Enhanced RAG with improved retrieval techniques
- [Simple Agent](simple-agent.md) - Agent that can use tools to solve complex tasks

## Running the Examples

You can run any of the examples using the provided script:

```bash
./run-examples.sh <example-name>
```

For example, to run the SimpleChat example:

```bash
export OPENAI_API_KEY=your-api-key
./run-examples.sh SimpleChat
```

You can also run the examples directly with SBT:

```bash
export OPENAI_API_KEY=your-api-key
sbt "examples/runMain zio.langchain.examples.SimpleChat"
```

## Example Complexity

The examples are organized in increasing order of complexity:

1. **Simple Chat**: Basic LLM interaction with a simple prompt
2. **Simple RAG**: Adding document retrieval to enhance responses
3. **Advanced Chat**: Adding streaming, memory, and more sophisticated interaction
4. **Enhanced RAG**: More advanced retrieval techniques and prompt engineering
5. **Simple Agent**: Full agent capabilities with tool usage

## Learning Path

For beginners, we recommend following this sequence:

1. Start with [Simple Chat](simple-chat.md) to understand basic LLM interaction
2. Move to [Simple RAG](simple-rag.md) to learn about embeddings and retrieval
3. Try [Advanced Chat](advanced-chat.md) to explore streaming and memory
4. Experiment with [Enhanced RAG](enhanced-rag.md) for more sophisticated retrieval
5. Finally, explore [Simple Agent](simple-agent.md) to understand agent capabilities

Each example builds on concepts from previous ones, creating a natural learning progression.