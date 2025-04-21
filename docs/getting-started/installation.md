---
title: Installation
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Installation

This guide will walk you through the process of adding ZIO LangChain to your Scala project.

## Prerequisites

Before installing ZIO LangChain, ensure you have the following:

- Scala 3.3.1+
- SBT 1.10.11+
- Java 11+

## Adding Dependencies

Add the following dependencies to your `build.sbt`:

```scala
val zioLangchainVersion = "0.1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "dev.zio" %% "zio-langchain-core" % zioLangchainVersion,
  "dev.zio" %% "zio-langchain-openai" % zioLangchainVersion
)
```

### Additional Modules

Depending on your needs, you may want to add more modules:

```scala
libraryDependencies ++= Seq(
  // Memory implementation
  "dev.zio" %% "zio-langchain-memory" % zioLangchainVersion,
  
  // Document loaders and parsers
  "dev.zio" %% "zio-langchain-document-loaders" % zioLangchainVersion,
  "dev.zio" %% "zio-langchain-document-parsers" % zioLangchainVersion,
  
  // Retrieval components
  "dev.zio" %% "zio-langchain-retrievers" % zioLangchainVersion,
  
  // Chain implementations
  "dev.zio" %% "zio-langchain-chains" % zioLangchainVersion,
  
  // Agent implementations
  "dev.zio" %% "zio-langchain-agents" % zioLangchainVersion,
  
  // Tool implementations
  "dev.zio" %% "zio-langchain-tools" % zioLangchainVersion
)
```

## Alternative Model Providers

If you want to use a different model provider instead of OpenAI, add the appropriate integration:

```scala
libraryDependencies ++= Seq(
  // Anthropic integration
  "dev.zio" %% "zio-langchain-anthropic" % zioLangchainVersion,
  
  // HuggingFace integration
  "dev.zio" %% "zio-langchain-huggingface" % zioLangchainVersion
)
```

## Next Steps

Once you have added the dependencies, you can proceed to:

1. [Configure](configuration.md) your API keys and model settings
2. Follow the [Quick Start Guide](quickstart.md) to build your first application