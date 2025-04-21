# ZIO LangChain

A comprehensive Scala 3 library that provides a ZIO-based langchain, offering a purely functional effect based, type-safe API for building LLM-powered applications.

## Features

- **Pure Functional**: All operations are represented as ZIO effects
- **Type-Safe**: Leverage Scala 3's type system for safer code
- **Composable**: Build complex workflows by composing simple components
- **Resource-Safe**: Proper resource management via ZIO
- **Streaming Support**: Stream tokens and process large documents efficiently
- **Comprehensive**: Support for all major langchain4j features

## Modules

- **Core**: Fundamental abstractions and interfaces
- **Models**: LLM implementations (OpenAI, Anthropic, etc.)
- **Embeddings**: Embedding model implementations
- **Memory**: Conversation memory implementations
- **Document Loaders**: Utilities for loading documents from various sources
- **Document Parsers**: Utilities for parsing and chunking documents
- **Retrievers**: Document retrieval implementations
- **Chains**: Composable operations for building workflows
- **Agents**: Autonomous systems that can use tools and solve tasks
- **Tools**: Implementations of various tools for agents

## Getting Started

### Prerequisites

- Scala 3.3.1+
- SBT 1.10.11+
- Java 11+

### Installation

Add the following dependencies to your `build.sbt`:

```scala
val zioLangchainVersion = "0.1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "dev.zio" %% "zio-langchain-core" % zioLangchainVersion,
  "dev.zio" %% "zio-langchain-openai" % zioLangchainVersion
)
```

### Configuration

Set the required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-3.5-turbo"  # Optional, defaults to gpt-3.5-turbo
export OPENAI_TEMPERATURE="0.7"      # Optional, defaults to 0.7
```

Or configure programmatically:

```scala
val openAIConfig = OpenAIConfig(
  apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
  model = "gpt-3.5-turbo",
  temperature = 0.7
)
```

## Quick Start Guide

This guide will help you quickly build and run the example applications in this project.

### Building the Project

To build the entire project, run:

```bash
sbt compile
```

This will compile all modules and prepare them for execution.

### Running the Examples

We provide a convenient script to run the example applications. First, make the script executable:

```bash
chmod +x run-examples.sh
```

Then you can run any of the examples with:

```bash
./run-examples.sh <example-name>
```

Available examples:
- `SimpleChat` - Basic chat application using ZIO LangChain
- `SimpleRAG` - Retrieval-Augmented Generation with document retrieval
- `AdvancedChat` - Chat with streaming responses and memory capabilities
- `EnhancedRAG` - Enhanced RAG with improved retrieval techniques
- `SimpleAgent` - Agent that can use tools to complete tasks

Example usage:
```bash
OPENAI_API_KEY=your-api-key ./run-examples.sh SimpleChat
```

To see all available examples and run them interactively, run the script without arguments:
```bash
./run-examples.sh
```

### Required Environment Variables

The examples require an OpenAI API key, which should be set as an environment variable:

```bash
export OPENAI_API_KEY=your-openai-api-key
```

If the API key is not set, the run script will display a warning and exit.

### Example Applications

1. **SimpleChat**: A basic chat application with an LLM.
   - Demonstrates basic LLM interaction
   - Uses a simple in-memory conversation history

2. **SimpleRAG**: A Retrieval-Augmented Generation example.
   - Loads and processes documents
   - Creates embeddings for document chunks
   - Retrieves relevant documents for queries
   - Generates responses based on retrieved context

3. **AdvancedChat**: An enhanced chat application with additional features.
   - Supports streaming responses (tokens appear one by one)
   - Includes function/tool calling capabilities
   - Uses BufferMemory for conversation history

4. **EnhancedRAG**: An improved RAG implementation.
   - Enhanced document processing and chunking
   - Improved similarity search for better retrieval
   - More sophisticated prompt engineering

5. **SimpleAgent**: An agent that can use tools to solve tasks.
   - Implements ReAct (Reasoning, Acting, Observing) pattern
   - Includes a calculator tool and a mock search tool
   - Makes decisions about which tools to use
   - Handles multi-step reasoning

## Examples

### Simple Chat

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object ChatExample extends ZIOAppDefault:
  override def run =
    for
      llm <- ZIO.service[LLM]
      response <- llm.complete("Hello, how are you?")
      _ <- Console.printLine(s"AI: $response")
    yield ()
    .provide(
      OpenAILLM.live,
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo"
        )
      )
    )
```

### Retrieval-Augmented Generation (RAG)

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object RAGExample extends ZIOAppDefault:
  override def run =
    for
      // Load documents
      documents <- loadDocuments()
      
      // Get embedding model
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Create embeddings
      embeddedDocs <- embeddingModel.embedDocuments(documents)
      
      // Create retriever
      retriever = createRetriever(embeddedDocs)
      
      // Get LLM
      llm <- ZIO.service[LLM]
      
      // Query
      query = "What is ZIO?"
      relevantDocs <- retriever.retrieve(query)
      context = relevantDocs.map(_.content).mkString("\n\n")
      prompt = s"Based on the following context:\n\n$context\n\nAnswer: $query"
      response <- llm.complete(prompt)
      
      // Print response
      _ <- Console.printLine(response)
    yield ()
    .provide(
      OpenAILLM.live,
      OpenAIEmbedding.live,
      ZLayer.succeed(OpenAIConfig(...)),
      ZLayer.succeed(OpenAIEmbeddingConfig(...))
    )
```

## Advanced Usage

### Creating a Chain

```scala
import zio.langchain.core.chain.*

// Create a chain that retrieves documents
val retrievalChain = Chain[Any, Throwable, String, Seq[Document]] { query =>
  retriever.retrieve(query).mapError(e => e.cause)
}

// Create a chain that formats a prompt
val promptChain = Chain[Any, Throwable, Seq[Document], String] { docs =>
  val context = docs.map(_.content).mkString("\n\n")
  ZIO.succeed(s"Context:\n$context\n\nQuestion: {{question}}\n\nAnswer:")
}

// Create a chain that replaces placeholders
val templateChain = Chain[Any, Throwable, (String, String), String] { 
  case (template, question) =>
    ZIO.succeed(template.replace("{{question}}", question))
}

// Create a chain that generates a response
val llmChain = Chain[Any, Throwable, String, String] { prompt =>
  llm.complete(prompt).mapError(e => e)
}

// Compose the chains
val answerChain = retrievalChain >>> promptChain >>> templateChain >>> llmChain

// Use the chain
val result = answerChain.run("What is ZIO?")
```

### Using Agents with Tools

```scala
import zio.langchain.core.agent.*
import zio.langchain.core.tool.*

// Create tools
val calculatorTool = Tool.make("calculator", "Calculate mathematical expressions") { input =>
  // Parse and evaluate the mathematical expression
  ZIO.attempt {
    val expression = input.trim
    val result = /* calculate result */
    result.toString
  }
}

val searchTool = Tool.make("search", "Search for information") { query =>
  ZIO.succeed(s"Search results for: $query")
}

// Create an agent
val agent = Agent.builder
  .tools(calculatorTool, searchTool)
  .llm(llm)
  .build

// Use the agent
val result = agent.run("What is 25 * 76 + 14?")
```

## License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE file for details.