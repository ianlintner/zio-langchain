---
title: LLM Integration
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# LLM Integration

This document explains how Large Language Models (LLMs) are integrated in ZIO LangChain - the core abstractions, available integrations, and patterns for interacting with language models.

## Table of Contents

- [Introduction](#introduction)
- [Core LLM Interface](#core-llm-interface)
- [Available Integrations](#available-integrations)
- [Usage Patterns](#usage-patterns)
- [Streaming Support](#streaming-support)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Introduction

Large Language Models (LLMs) are the foundation of ZIO LangChain. They provide the intelligence and natural language capabilities that power all other components. ZIO LangChain offers a clean, ZIO-native interface to interact with different LLM providers, including:

- OpenAI (GPT models)
- Anthropic (Claude models)
- HuggingFace (open-source models)
- Local models via ollama or llama.cpp

The library abstracts away the provider-specific details, offering a unified interface while preserving ZIO's functional approach and benefits:

- Pure functional implementation with ZIO effects
- Proper resource management
- Structured error handling
- Composability with the ZIO ecosystem
- Type safety throughout

## Core LLM Interface

ZIO LangChain defines a core `LLM` trait that represents the fundamental capabilities of language models:

```scala
trait LLM:
  def complete(prompt: String): ZIO[Any, LLMError, String]
  
  def completeChat(messages: Seq[ChatMessage]): ZIO[Any, LLMError, ChatResponse]
  
  def tokenize(text: String): ZIO[Any, LLMError, Seq[String]]
  
  def countTokens(text: String): ZIO[Any, LLMError, Int]
```

This interface provides:

- **Text completion**: Generate a text response given a prompt
- **Chat completion**: Generate a response given a sequence of chat messages
- **Tokenization**: Split text into tokens as processed by the model
- **Token counting**: Count tokens without generating responses (useful for context management)

All operations return `ZIO` effects, propagating errors through the error channel and enabling composition with other ZIO operations.

## Available Integrations

### OpenAI Integration

The OpenAI integration provides access to GPT models:

```scala
import zio.*
import zio.langchain.integrations.openai.*
import zio.langchain.core.model.LLM

val program = for {
  llm <- ZIO.service[LLM]
  response <- llm.complete("Explain ZIO in one sentence.")
  _ <- Console.printLine(s"Response: $response")
} yield ()

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

#### Chat Interface

OpenAI models can be used with the chat interface:

```scala
import zio.langchain.core.domain.*

val chatProgram = for {
  llm <- ZIO.service[LLM]
  messages = Seq(
    ChatMessage(Role.System, "You are a helpful assistant."),
    ChatMessage(Role.User, "Explain functional programming in Scala.")
  )
  response <- llm.completeChat(messages)
  _ <- Console.printLine(s"Response: ${response.message.content}")
} yield ()
```

### Anthropic Integration

The Anthropic integration provides access to Claude models:

```scala
import zio.*
import zio.langchain.integrations.anthropic.*
import zio.langchain.core.model.LLM

val program = for {
  llm <- ZIO.service[LLM]
  response <- llm.complete("Explain ZIO in one sentence.")
  _ <- Console.printLine(s"Response: $response")
} yield ()

program.provide(
  AnthropicLLM.live,
  ZLayer.succeed(
    AnthropicConfig(
      apiKey = sys.env.getOrElse("ANTHROPIC_API_KEY", ""),
      model = "claude-2"
    )
  )
)
```

### HuggingFace Integration

The HuggingFace integration provides access to various open-source models:

```scala
import zio.*
import zio.langchain.integrations.huggingface.*
import zio.langchain.core.model.LLM

val program = for {
  llm <- ZIO.service[LLM]
  response <- llm.complete("Explain ZIO in one sentence.")
  _ <- Console.printLine(s"Response: $response")
} yield ()

program.provide(
  HuggingFaceLLM.live,
  ZLayer.succeed(
    HuggingFaceConfig(
      apiKey = sys.env.getOrElse("HF_API_KEY", ""),
      model = "mistralai/Mistral-7B-Instruct-v0.1"
    )
  )
)
```

## Usage Patterns

### Basic Prompt Completion

The simplest usage pattern is direct prompt completion:

```scala
val result = llm.complete("Explain quantum computing.")
```

### Templated Prompts

For more complex prompts, you can use template processing:

```scala
import zio.langchain.core.prompt.PromptTemplate

val template = PromptTemplate(
  template = "Explain {topic} in {style} style with {num_sentences} sentences.",
  inputVariables = Set("topic", "style", "num_sentences")
)

val variables = Map(
  "topic" -> "quantum computing", 
  "style" -> "simple", 
  "num_sentences" -> "3"
)

val program = for {
  prompt <- ZIO.attempt(template.format(variables))
                .mapError(e => LLMError(e))
  response <- llm.complete(prompt)
} yield response
```

### Chat Conversations

For multi-turn conversations:

```scala
import zio.langchain.core.domain.*

val messages = Seq(
  ChatMessage(Role.System, "You are a helpful technical assistant specialized in Scala programming."),
  ChatMessage(Role.User, "How do I create a simple web server in Scala?"),
  ChatMessage(Role.Assistant, "You can use libraries like http4s, Play Framework, or ZIO HTTP to create a web server in Scala."),
  ChatMessage(Role.User, "Can you show me a simple example using ZIO HTTP?")
)

val chatProgram = for {
  llm <- ZIO.service[LLM]
  response <- llm.completeChat(messages)
} yield response.message.content
```

### Token Management

Monitor and manage token usage:

```scala
val tokenManagementProgram = for {
  llm <- ZIO.service[LLM]
  
  // Count tokens in a prompt
  prompt = "This is a long prompt that I want to check for token count..."
  tokenCount <- llm.countTokens(prompt)
  
  // Only proceed if under limit
  response <- ZIO.when(tokenCount < 1000) {
    llm.complete(prompt)
  }
} yield response
```

## Streaming Support

ZIO LangChain supports streaming responses for real-time interaction:

```scala
import zio.*
import zio.stream.*
import zio.langchain.core.model.StreamingLLM
import zio.langchain.integrations.openai.*

val streamingProgram = for {
  llm <- ZIO.service[StreamingLLM]
  
  // Create streaming effect
  stream <- llm.completeStreaming("Generate a short story about a robot.")
  
  // Process the stream
  _ <- stream.foreach { chunk =>
    Console.print(chunk).orDie
  }
} yield ()

streamingProgram.provide(
  OpenAILLM.streaming,
  ZLayer.succeed(
    OpenAIConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "gpt-3.5-turbo",
      streaming = true
    )
  )
)
```

### Chat Streaming

Chat conversations can also be streamed:

```scala
val streamingChatProgram = for {
  llm <- ZIO.service[StreamingLLM]
  
  messages = Seq(
    ChatMessage(Role.System, "You are a creative storyteller."),
    ChatMessage(Role.User, "Tell me a short story about space exploration.")
  )
  
  stream <- llm.completeChatStreaming(messages)
  
  _ <- stream.foreach { chunk =>
    Console.print(chunk).orDie
  }
} yield ()
```

## Error Handling

ZIO LangChain provides a structured error hierarchy for LLM interactions:

```scala
sealed trait LLMError extends Throwable:
  def message: String
  def cause: Throwable

case class ProviderError(message: String, cause: Throwable = null) extends LLMError
case class RateLimitError(message: String, cause: Throwable = null) extends LLMError
case class InvalidRequestError(message: String, cause: Throwable = null) extends LLMError
case class AuthenticationError(message: String, cause: Throwable = null) extends LLMError
case class ServiceUnavailableError(message: String, cause: Throwable = null) extends LLMError
```

This enables fine-grained error handling:

```scala
val robustProgram = llm.complete("Tell me about ZIO.")
  .catchSome {
    case e: RateLimitError => 
      // Wait and retry
      ZIO.logWarning(s"Rate limit hit: ${e.message}") *>
      ZIO.sleep(2.seconds) *>
      llm.complete("Tell me about ZIO.")
      
    case e: ServiceUnavailableError =>
      // Use fallback
      ZIO.logWarning(s"Service unavailable: ${e.message}") *>
      fallbackLLM.complete("Tell me about ZIO.")
  }
```

## Best Practices

### Configuration Management

Use environment variables for sensitive configuration:

```scala
val config = OpenAIConfig(
  apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
  model = sys.env.getOrElse("OPENAI_MODEL", "gpt-3.5-turbo")
)
```

Or use ZIO Config for more robust configuration:

```scala
import zio.config.*
import zio.config.magnolia.*
import zio.config.typesafe.*

case class LLMConfig(
  apiKey: String,
  model: String,
  temperature: Double,
  maxTokens: Option[Int]
)

val configLayer = ZLayer.fromZIO(
  TypesafeConfigSource.fromResourcePath
    .flatMap(source => descriptor[LLMConfig].from(source))
)
```

### Performance Optimization

Batch similar requests when possible:

```scala
val batchProgram = for {
  llm <- ZIO.service[LLM]
  
  prompts = Seq(
    "Summarize quantum computing",
    "Explain neural networks",
    "Describe blockchains"
  )
  
  // Process in parallel with rate limiting
  responses <- ZIO.foreachPar(prompts)(prompt => 
    llm.complete(prompt)
  ).withParallelism(3) // Limit concurrent requests
} yield responses
```

### Prompt Engineering

Follow best practices for prompt design:

```scala
val wellDesignedPrompt = """
You are an expert Scala developer. 
Your task is to help write clean, functional Scala code.

CONTEXT:
The user is building a web application using ZIO and wants to implement proper error handling.

QUESTION:
What's the best way to handle HTTP errors in a ZIO HTTP application?

Please provide:
1. A brief explanation of ZIO error handling concepts
2. Concrete code examples
3. Best practices for HTTP error responses
"""

llm.complete(wellDesignedPrompt)
```

### Provider Resilience

Create a resilient LLM service with fallbacks:

```scala
import zio.*
import zio.langchain.core.model.*

// Create a resilient LLM with fallback
def createResilientLLM(
  primaryLLM: LLM,
  fallbackLLM: LLM
): LLM = new LLM {
  override def complete(prompt: String): ZIO[Any, LLMError, String] =
    primaryLLM.complete(prompt)
      .catchAll { error =>
        ZIO.logWarning(s"Primary LLM failed: ${error.message}. Using fallback...") *>
        fallbackLLM.complete(prompt)
      }
      
  // Implement other methods with similar pattern
  override def completeChat(messages: Seq[ChatMessage]): ZIO[Any, LLMError, ChatResponse] =
    primaryLLM.completeChat(messages)
      .catchAll { error =>
        ZIO.logWarning(s"Primary LLM failed: ${error.message}. Using fallback...") *>
        fallbackLLM.completeChat(messages)
      }
  
  override def tokenize(text: String): ZIO[Any, LLMError, Seq[String]] =
    primaryLLM.tokenize(text)
      .catchAll { error =>
        ZIO.logWarning(s"Primary LLM failed: ${error.message}. Using fallback...") *>
        fallbackLLM.tokenize(text)
      }
  
  override def countTokens(text: String): ZIO[Any, LLMError, Int] =
    primaryLLM.countTokens(text)
      .catchAll { error =>
        ZIO.logWarning(s"Primary LLM failed: ${error.message}. Using fallback...") *>
        fallbackLLM.countTokens(text)
      }
}

// Usage
val openAILLM = OpenAILLM.make(openAIConfig)
val anthropicLLM = AnthropicLLM.make(anthropicConfig)
val resilientLLM = createResilientLLM(openAILLM, anthropicLLM)