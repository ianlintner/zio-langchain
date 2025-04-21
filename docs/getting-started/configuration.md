---
title: Configuration Guide
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Configuration Guide

This guide explains how to configure ZIO LangChain to work with various LLM providers and customize behavior for your application.

## Table of Contents

- [General Configuration Principles](#general-configuration-principles)
- [OpenAI Configuration](#openai-configuration)
- [Anthropic Configuration](#anthropic-configuration)
- [HuggingFace Configuration](#huggingface-configuration)
- [Environment Variables & Secrets](#environment-variables--secrets)
- [Configuration by Environment](#configuration-by-environment)
- [Advanced Configuration](#advanced-configuration)

## General Configuration Principles

ZIO LangChain uses ZIO's layer system for dependency injection and configuration. Each integration provides a `live` ZLayer that can be configured with appropriate settings.

The general pattern is:

```scala
import zio.*
import zio.langchain.integrations.{provider}.*

// Create the layer
val providerLayer = ProviderService.live.provide(
  ZLayer.succeed(
    ProviderConfig(
      // Configuration parameters
    )
  )
)

// Use in your application
val app = for {
  service <- ZIO.service[ServiceType]
  result <- service.operation(...)
} yield result

// Provide the layer
app.provide(providerLayer)
```

## OpenAI Configuration

To configure OpenAI integration:

```scala
import zio.*
import zio.langchain.integrations.openai.*

// Basic configuration
val openAILayer = OpenAILLM.live.provide(
  ZLayer.succeed(
    OpenAIConfig(
      apiKey = "your-api-key",
      model = "gpt-3.5-turbo",
      temperature = 0.7,
      maxTokens = 2000
    )
  )
)

// For embeddings
val openAIEmbeddingLayer = OpenAIEmbedding.live.provide(
  ZLayer.succeed(
    OpenAIEmbeddingConfig(
      apiKey = "your-api-key",
      model = "text-embedding-ada-002"
    )
  )
)
```

### OpenAI Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `apiKey` | Your OpenAI API key | (Required) | `"sk-..."` |
| `model` | Model to use | `"gpt-3.5-turbo"` | `"gpt-4"` |
| `temperature` | Controls randomness (0.0-2.0) | `0.7` | `0.0` for deterministic |
| `maxTokens` | Maximum tokens to generate | `None` | `2000` |
| `topP` | Nucleus sampling parameter | `1.0` | `0.9` |
| `presencePenalty` | Penalty for token presence | `0.0` | `0.5` |
| `frequencyPenalty` | Penalty for token frequency | `0.0` | `0.5` |
| `stop` | Sequences to stop generation | `None` | `Seq("\n\n")` |
| `timeout` | Timeout duration | `30.seconds` | `1.minute` |
| `apiHost` | Alternative API host | `"api.openai.com"` | For proxy configurations |

## Anthropic Configuration

To configure Anthropic integration:

```scala
import zio.*
import zio.langchain.integrations.anthropic.*

val anthropicLayer = AnthropicLLM.live.provide(
  ZLayer.succeed(
    AnthropicConfig(
      apiKey = "your-anthropic-api-key",
      model = "claude-2",
      temperature = 0.5,
      maxTokens = 1000
    )
  )
)
```

### Anthropic Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `apiKey` | Your Anthropic API key | (Required) | `"sk-ant-..."` |
| `model` | Model to use | `"claude-2"` | `"claude-instant-1"` |
| `temperature` | Controls randomness (0.0-1.0) | `0.5` | `0.0` for deterministic |
| `maxTokens` | Maximum tokens to generate | `1000` | `2000` |
| `topP` | Nucleus sampling parameter | `1.0` | `0.9` |
| `topK` | Top-K sampling parameter | `None` | `40` |
| `timeout` | Timeout duration | `30.seconds` | `1.minute` |

## HuggingFace Configuration

To configure HuggingFace integration:

```scala
import zio.*
import zio.langchain.integrations.huggingface.*

val huggingFaceLayer = HuggingFaceLLM.live.provide(
  ZLayer.succeed(
    HuggingFaceConfig(
      apiKey = "your-hf-api-key",
      model = "mistralai/Mistral-7B-Instruct-v0.1",
      temperature = 0.7,
      maxLength = 512
    )
  )
)

val huggingFaceEmbeddingLayer = HuggingFaceEmbedding.live.provide(
  ZLayer.succeed(
    HuggingFaceEmbeddingConfig(
      apiKey = "your-hf-api-key",
      model = "sentence-transformers/all-MiniLM-L6-v2"
    )
  )
)
```

### HuggingFace Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `apiKey` | Your HuggingFace API key | (Required) | `"hf_..."` |
| `model` | Model to use | (Required) | `"mistralai/Mistral-7B-Instruct-v0.1"` |
| `temperature` | Controls randomness | `0.7` | `0.0` for deterministic |
| `maxLength` | Maximum tokens to generate | `512` | `1024` |
| `topP` | Nucleus sampling parameter | `0.9` | `0.8` |
| `topK` | Top-K sampling parameter | `50` | `40` |
| `repetitionPenalty` | Penalty for repetitions | `1.0` | `1.2` |
| `timeout` | Timeout duration | `30.seconds` | `1.minute` |

## Environment Variables & Secrets

It's best practice to store API keys in environment variables rather than hardcoding them:

```scala
// Load from environment variable
val openAIKey = sys.env.getOrElse("OPENAI_API_KEY", "")

// Or use ZIO Config for more robust configuration management
import zio.config.*
import zio.config.magnolia.*
import zio.config.typesafe.*

case class AppConfig(
  openAiApiKey: String,
  openAiModel: String,
  temperature: Double
)

val appConfig = descriptor[AppConfig].map(TypesafeConfigSource.fromResourcePath)

val configLayer = ZLayer.fromZIO(appConfig)

// Use in layers
val openAILayer = ZLayer.service[AppConfig].flatMap(config => 
  OpenAILLM.live.provide(
    ZLayer.succeed(
      OpenAIConfig(
        apiKey = config.get.openAiApiKey,
        model = config.get.openAiModel,
        temperature = config.get.temperature
      )
    )
  )
)
```

## Configuration by Environment

You can implement environment-specific configuration:

```scala
import zio.*

// Base configuration
val baseConfig = OpenAIConfig(
  apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
  model = "gpt-3.5-turbo",
  temperature = 0.7
)

// Environment-specific configurations
val devConfig = baseConfig.copy(
  model = "gpt-3.5-turbo",
  timeout = 1.minute
)

val prodConfig = baseConfig.copy(
  model = "gpt-4",
  temperature = 0.5,
  maxTokens = Some(1000),
  timeout = 30.seconds
)

// Select based on environment
val environment = sys.env.getOrElse("APP_ENV", "dev")
val config = environment match {
  case "prod" => prodConfig
  case "staging" => prodConfig.copy(model = "gpt-3.5-turbo")
  case _ => devConfig
}

val openAILayer = OpenAILLM.live.provide(
  ZLayer.succeed(config)
)
```

## Advanced Configuration

### Proxy Configuration

For organizations that require a proxy for API access:

```scala
val proxyConfig = OpenAIConfig(
  apiKey = "your-api-key",
  model = "gpt-3.5-turbo",
  apiHost = "your-proxy-host.internal",
  headers = Map(
    "X-Corporate-Proxy" -> "enabled",
    "X-Internal-User" -> "service-account"
  )
)
```

### Streaming Configuration

Configure streaming responses:

```scala
val streamingConfig = OpenAIConfig(
  apiKey = "your-api-key",
  model = "gpt-3.5-turbo",
  streaming = true,
  streamingChunkSize = 20, // tokens per chunk
  streamingBufferSize = 100 // buffer size for the streaming queue
)
```

### Rate Limiting

Implement rate limiting for API calls:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.integrations.openai.*

// Create a rate-limited LLM service
def rateLimitedLLM(
  config: OpenAIConfig, 
  maxCalls: Int, 
  interval: Duration
): ZLayer[Any, Nothing, LLM] = {
  val semaphore = Semaphore.make(maxCalls)
  val throttled = ZLayer.fromZIO(semaphore).flatMap { sem =>
    val rateLimitedLLM = new LLM {
      val underlying = OpenAILLM.make(config)
      
      override def complete(prompt: String): ZIO[Any, LLMError, String] =
        sem.get.withPermit {
          for {
            result <- underlying.complete(prompt)
            _ <- ZIO.sleep(interval / maxCalls)
          } yield result
        }
        
      // Implement other methods
    }
    
    ZLayer.succeed(rateLimitedLLM)
  }
  
  throttled
}

// Use the rate-limited service
val limitedLayer = rateLimitedLLM(
  OpenAIConfig(apiKey = "your-api-key", model = "gpt-4"),
  maxCalls = 60,
  interval = 1.minute
)
```

### Fallback Configuration

Configure fallbacks for robustness:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.integrations.openai.*
import zio.langchain.integrations.anthropic.*

// Primary service
val primaryLLM = OpenAILLM.live.provide(
  ZLayer.succeed(OpenAIConfig(apiKey = "openai-key", model = "gpt-4"))
)

// Fallback service
val fallbackLLM = AnthropicLLM.live.provide(
  ZLayer.succeed(AnthropicConfig(apiKey = "anthropic-key", model = "claude-2"))
)

// Create a fallback LLM service
val fallbackLayer = ZLayer.fromZIO(
  ZIO.environment[LLM].flatMap { primary =>
    ZIO.environment[LLM].map { fallback =>
      new LLM {
        override def complete(prompt: String): ZIO[Any, LLMError, String] =
          primary.get.complete(prompt)
            .catchAll { error =>
              Console.printLine(s"Primary LLM failed: ${error.message}. Trying fallback...").orDie *>
              fallback.get.complete(prompt)
            }
      }
    }.provideLayer(fallbackLLM)
  }.provideLayer(primaryLLM)
)