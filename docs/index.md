---
title: ZIO LangChain Documentation
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# ZIO LangChain

Welcome to the official documentation for ZIO LangChain, a Scala library that provides ZIO-based bindings for langchain4j, enabling you to build robust, typesafe LLM applications with ZIO.

## Overview

ZIO LangChain combines the power of [ZIO](https://zio.dev/) - a type-safe, composable library for asynchronous and concurrent programming in Scala - with the capabilities of langchain4j, creating a purely functional and type-safe API for building Large Language Model (LLM) applications.

```ascii
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                       ZIO LangChain                         │
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐  │
│  │   LLMs  │   │ Chains  │   │ Agents  │   │ Retrievers  │  │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘  │
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐  │
│  │ Memory  │   │  Tools  │   │  Docs   │   │ Embeddings  │  │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘  │
│                                                             │
│           Built on ZIO for pure functional LLMs             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Pure Functional**: All operations are represented as ZIO effects
- **Type Safety**: Strong type safety throughout the API
- **Composability**: Easy composition of components via ZIO
- **Resource Safety**: Proper resource management via ZIO
- **Streaming**: Support for streaming responses
- **Concurrency**: Powerful concurrency patterns via ZIO
- **Error Handling**: Comprehensive error hierarchy and handling

## Documentation Structure

The documentation is organized into the following sections:

- **[Getting Started](getting-started/index.md)**: Installation, configuration, and quick start guides
- **[Core Concepts](core-concepts/index.md)**: Detailed explanations of the main abstractions
- **[Examples](examples/index.md)**: Practical examples demonstrating library usage
- **[Design](design/architecture.md)**: Architectural overview and design decisions
- **API Reference**: Comprehensive API documentation

## Quick Navigation

| If you want to... | Go to... |
|-------------------|----------|
| Install the library | [Installation Guide](getting-started/installation.md) |
| Configure providers | [Configuration Guide](getting-started/configuration.md) |
| Get started quickly | [Quick Start Guide](getting-started/quickstart.md) |
| Understand core concepts | [Core Concepts](core-concepts/index.md) |
| See practical examples | [Examples](examples/index.md) |
| Learn about the architecture | [Architecture Overview](design/architecture.md) |

## Getting Started

To start using ZIO LangChain, add the following dependencies to your build.sbt:

```scala
// Core library
libraryDependencies += "dev.zio" %% "zio-langchain-core" % "0.1.0"

// OpenAI integration
libraryDependencies += "dev.zio" %% "zio-langchain-openai" % "0.1.0"

// Optional modules
libraryDependencies += "dev.zio" %% "zio-langchain-chains" % "0.1.0"
libraryDependencies += "dev.zio" %% "zio-langchain-agents" % "0.1.0"
```

Then, set up your OpenAI API key:

```scala
import zio.*
import zio.langchain.integrations.openai.*

// Using environment variable
val openAILayer = OpenAILLM.live.provide(
  ZLayer.succeed(
    OpenAIConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "gpt-3.5-turbo"
    )
  )
)
```

Create a simple application:

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.integrations.openai.*

object SimpleApp extends ZIOAppDefault {
  val program = for {
    llm <- ZIO.service[LLM]
    response <- llm.complete("Explain functional programming in one sentence.")
    _ <- Console.printLine(s"Response: $response")
  } yield ()
  
  override def run = program.provide(
    OpenAILLM.live,
    ZLayer.succeed(OpenAIConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "gpt-3.5-turbo"
    ))
  )
}
```

For more comprehensive examples, see the [Examples](examples/index.md) section.

## Examples

Here are a few examples of what you can build with ZIO LangChain:

- Simple chat applications
- Question answering systems with document retrieval (RAG)
- Autonomous agents that use tools to solve tasks
- Conversational agents with memory
- Document processing and analysis pipelines

Check out the [Examples](examples/index.md) section for complete code samples.

## Contributing

Contributions to ZIO LangChain are welcome! Whether it's bug reports, feature requests, or code contributions, please feel free to contribute through GitHub issues and pull requests.

## License

ZIO LangChain is licensed under the Apache License 2.0.