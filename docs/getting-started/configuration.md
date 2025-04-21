---
title: Configuration
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Configuration

This guide explains how to configure ZIO LangChain to work with various model providers.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Programmatic Configuration](#programmatic-configuration)
- [ZIO Config Integration](#zio-config-integration)
- [Configuration Options](#configuration-options)
- [Integration-Specific Configuration](#integration-specific-configuration)

## Environment Variables

The simplest way to configure ZIO LangChain is through environment variables:

```bash
# OpenAI configuration
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-3.5-turbo"  # Optional, defaults to gpt-3.5-turbo
export OPENAI_TEMPERATURE="0.7"      # Optional, defaults to 0.7

# Anthropic configuration (if using the Anthropic integration)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_MODEL="claude-2"    # Optional, defaults to claude-2
```

## Programmatic Configuration

You can also configure ZIO LangChain programmatically:

```scala
import zio.langchain.integrations.openai.OpenAIConfig

val openAIConfig = OpenAIConfig(
  apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
  model = "gpt-3.5-turbo",
  temperature = 0.7,
  maxTokens = Some(2000),
  organizationId = None,
  timeout = 60.seconds
)
```

Then provide this configuration to your application:

```scala
myApp.provide(
  OpenAILLM.live,
  ZLayer.succeed(openAIConfig)
)
```

## ZIO Config Integration

ZIO LangChain uses ZIO Config for type-safe configuration handling. You can load configuration from various sources:

```scala
import zio.*
import zio.config.*
import zio.config.typesafe.*
import zio.langchain.integrations.openai.OpenAIConfig

// Load from application.conf
val configLayer = TypesafeConfig.fromDefaultLoader[OpenAIConfig](
  ConfigDescriptor.descriptor[OpenAIConfig]
)

myApp.provide(
  OpenAILLM.live,
  configLayer
)
```

Example `application.conf`:

```hocon
openai {
  api-key = ${OPENAI_API_KEY}
  model = "gpt-4"
  temperature = 0.5
  max-tokens = 2000
  timeout = 120s
}
```

## Configuration Options

### Common Configuration Options

These options are available for most model integrations:

| Option | Description | Default |
|--------|-------------|---------|
| `apiKey` | API key for the model provider | Required |
| `model` | Model identifier to use | Depends on provider |
| `temperature` | Controls randomness (0.0-1.0) | 0.7 |
| `maxTokens` | Maximum tokens to generate | None (model default) |
| `timeout` | Request timeout | 60 seconds |

### Advanced Options

Some integrations support additional options:

| Option | Description | Default |
|--------|-------------|---------|
| `topP` | Nucleus sampling parameter | 1.0 |
| `presencePenalty` | Penalizes repeated tokens | 0.0 |
| `frequencyPenalty` | Penalizes frequent tokens | 0.0 |
| `stopSequences` | Sequences that stop generation | Empty |
| `logitBias` | Biases for specific tokens | Empty |

## Integration-Specific Configuration

### OpenAI

```scala
case class OpenAIConfig(
  apiKey: String,
  model: String = "gpt-3.5-turbo",
  temperature: Double = 0.7,
  maxTokens: Option[Int] = None,
  organizationId: Option[String] = None,
  timeout: Duration = 60.seconds
)
```

### Anthropic

```scala
case class AnthropicConfig(
  apiKey: String,
  model: String = "claude-2",
  temperature: Double = 0.7,
  maxTokens: Option[Int] = None,
  timeout: Duration = 60.seconds
)
```

### Embedding Models

```scala
case class OpenAIEmbeddingConfig(
  apiKey: String,
  model: String = "text-embedding-ada-002",
  organizationId: Option[String] = None,
  timeout: Duration = 60.seconds
)
```

## Next Steps

After configuring your models, proceed to the [Quick Start Guide](quickstart.md) to build your first application.