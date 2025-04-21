---
title: LLM Integration
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# LLM Integration

This document describes how ZIO LangChain integrates with Large Language Models (LLMs) and provides a unified interface for working with different model providers.

## Table of Contents

- [Overview](#overview)
- [Core LLM Interface](#core-llm-interface)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Usage Examples](#usage-examples)
- [Streaming Responses](#streaming-responses)
- [Error Handling](#error-handling)

## Overview

ZIO LangChain provides a consistent, type-safe interface to interact with various LLM providers. The library wraps provider-specific APIs in a ZIO-based interface, offering the benefits of ZIO's effect system, resource management, and error handling.

## Core LLM Interface

The foundation of LLM integration is the `LLM` trait:

```scala
trait LLM:
  def complete(prompt: String): ZIO[Any, LLMError, String]
  def completeChat(messages: Seq[ChatMessage]): ZIO[Any, LLMError, ChatResponse]
  
  // Streaming variants
  def streamComplete(prompt: String): ZStream[Any, LLMError, String]
  def streamCompleteChat(messages: Seq[ChatMessage]): ZStream[Any, LLMError, ChatResponse]
```

This interface provides two primary functionalities:

1. **Text Completion**: Generate text based on a single prompt
2. **Chat Completion**: Generate responses based on a conversation history

Both functionalities have streaming variants that return ZIO Streams for token-by-token responses.

## Key Features

- **ZIO Effect System**: All operations are represented as ZIO effects
- **Unified Interface**: Consistent API across different model providers
- **Type-Safe**: Leverage Scala 3's type system for safer code
- **Resource Management**: Proper resource handling via ZIO
- **Streaming Support**: Process model outputs token by token
- **Error Handling**: Comprehensive error representation and handling

## Supported Models

ZIO LangChain supports the following LLM providers:

- **OpenAI** (GPT-3.5, GPT-4)
- **Anthropic** (Claude models)
- **HuggingFace** (Various open source models)

Each integration is provided in a separate module with consistent interfaces.

## Usage Examples

### Basic Text Completion

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.integrations.openai.*

val program = for
  llm <- ZIO.service[LLM]
  response <- llm.complete("Explain quantum computing in simple terms.")
  _ <- Console.printLine(s"Response: $response")
yield ()

program.provide(
  OpenAILLM.live,
  ZLayer.succeed(
    OpenAIConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "gpt-3.5-turbo"
    )
  )
)
```

### Chat Completion

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

val messages = Seq(
  ChatMessage(Role.System, "You are a helpful assistant."),
  ChatMessage(Role.User, "What's the best way to learn Scala?")
)

val program = for
  llm <- ZIO.service[LLM]
  response <- llm.completeChat(messages)
  _ <- Console.printLine(s"AI: ${response.message.content}")
yield ()

program.provide(
  OpenAILLM.live,
  ZLayer.succeed(OpenAIConfig(...))
)
```

## Streaming Responses

For use cases where you want to process responses token by token (e.g., displaying a gradually appearing response), use the streaming variants:

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.console.Console

val program = for
  llm <- ZIO.service[LLM]
  _ <- llm.streamComplete("Write a short poem about programming.")
    .foreach(token => Console.printLine(token))
yield ()
```

This is particularly useful for:

1. Displaying responsive UIs where text appears gradually
2. Processing very long outputs in chunks
3. Implementing early stopping based on content

## Error Handling

ZIO LangChain uses the `LLMError` type for representing errors from LLM interactions:

```scala
case class LLMError(
  cause: Throwable,
  message: String = "LLM error occurred"
) extends LangChainError
```

You can handle these errors using ZIO's error handling mechanisms:

```scala
llm.complete("Your prompt")
  .catchAll { error =>
    // Handle the error
    Console.printLine(s"Error: ${error.message}, Cause: ${error.cause.getMessage}")
      .as("Fallback response")
  }
```

Common error scenarios include:

- API authentication failures
- Rate limiting or quota exceeded
- Timeout errors
- Malformed requests
- Content policy violations

For robust applications, always implement proper error handling to manage these potential failures.