---
title: Installation Guide
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Installation Guide

This guide explains how to add ZIO LangChain to your Scala project and set up the necessary dependencies.

## Table of Contents

- [SBT Setup](#sbt-setup)
- [Module Overview](#module-overview)
- [Minimum Requirements](#minimum-requirements)
- [Snapshot Versions](#snapshot-versions)
- [Cross-Building](#cross-building)
- [Verification](#verification)

## SBT Setup

Add the following dependencies to your `build.sbt` file:

### Core Module

The core module provides the fundamental abstractions and interfaces:

```scala
libraryDependencies += "dev.zio" %% "zio-langchain-core" % "0.1.0"
```

### Integration Modules

Add integration modules for specific LLM providers:

```scala
// OpenAI integration
libraryDependencies += "dev.zio" %% "zio-langchain-openai" % "0.1.0"

// Anthropic integration
libraryDependencies += "dev.zio" %% "zio-langchain-anthropic" % "0.1.0"

// HuggingFace integration
libraryDependencies += "dev.zio" %% "zio-langchain-huggingface" % "0.1.0"
```

### Additional Modules

Add functionality modules as needed:

```scala
// Chains module
libraryDependencies += "dev.zio" %% "zio-langchain-chains" % "0.1.0"

// Memory module
libraryDependencies += "dev.zio" %% "zio-langchain-memory" % "0.1.0"

// Retriever module
libraryDependencies += "dev.zio" %% "zio-langchain-retrievers" % "0.1.0"

// Agents module
libraryDependencies += "dev.zio" %% "zio-langchain-agents" % "0.1.0"

// Document loaders module
libraryDependencies += "dev.zio" %% "zio-langchain-document-loaders" % "0.1.0"

// Document parsers module
libraryDependencies += "dev.zio" %% "zio-langchain-document-parsers" % "0.1.0"

// Tools module
libraryDependencies += "dev.zio" %% "zio-langchain-tools" % "0.1.0"
```

## Module Overview

ZIO LangChain is organized into modular components:

| Module | Description |
|--------|-------------|
| `core` | Core abstractions and interfaces |
| `openai` | OpenAI model integrations |
| `anthropic` | Anthropic model integrations |
| `huggingface` | HuggingFace model integrations |
| `chains` | Chain implementations |
| `memory` | Memory implementations |
| `retrievers` | Retriever implementations |
| `agents` | Agent implementations |
| `document-loaders` | Document loading utilities |
| `document-parsers` | Document parsing utilities |
| `tools` | Tool definitions and implementations |

Each module can be included independently based on your needs.

## Minimum Requirements

ZIO LangChain requires:

- Scala 2.13 or 3.x
- JDK 11 or higher
- SBT 1.5.0 or higher
- ZIO 2.0.0 or higher

## Snapshot Versions

For the latest development features, you can use snapshot versions:

```scala
resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies += "dev.zio" %% "zio-langchain-core" % "0.2.0-SNAPSHOT"
```

## Cross-Building

ZIO LangChain supports cross-building for Scala 2.13 and 3.x:

```scala
// For Scala 2.13
scalaVersion := "2.13.10"
libraryDependencies += "dev.zio" %% "zio-langchain-core" % "0.1.0"

// For Scala 3.x
scalaVersion := "3.3.0"
libraryDependencies += "dev.zio" %% "zio-langchain-core" % "0.1.0"
```

## Verification

To verify that ZIO LangChain is correctly installed, you can create a simple test application:

```scala
import zio.*
import zio.Console.*
import zio.langchain.core.model.LLM
import zio.langchain.integrations.openai.*

object VerifyInstallation extends ZIOAppDefault {
  val program = for {
    _ <- printLine("Testing ZIO LangChain installation...")
    llm <- ZIO.service[LLM]
    response <- llm.complete("Hello, ZIO LangChain!")
    _ <- printLine(s"LLM Response: $response")
    _ <- printLine("Installation verified successfully!")
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

Run this application with your OpenAI API key set in the environment:

```bash
export OPENAI_API_KEY=your-api-key
sbt run
```

If everything is configured correctly, you should see a response from the LLM and a success message.