# ZIO LangChain Anthropic Integration

This module provides integration with Anthropic's Claude models for ZIO LangChain.

## Features

- Full implementation of the LLM trait for Anthropic Claude models
- Support for both simple text completion and chat completion
- Configuration via HOCON and environment variables
- Streaming support (planned for future implementation)
- Proper ZIO effect handling and error management

## Installation

Add the following dependency to your `build.sbt`:

```scala
libraryDependencies += "dev.zio" %% "zio-langchain-anthropic" % "0.1.0-SNAPSHOT"
```

## Configuration

The Anthropic integration can be configured using HOCON configuration or environment variables.

### HOCON Configuration

Create an `application.conf` file with the following structure:

```hocon
anthropic {
  # API key for Anthropic Claude API
  api-key = "your-api-key-here"
  api-key = ${?ANTHROPIC_API_KEY}
  
  # Model identifier to use
  model = "claude-3-sonnet-20240229"
  
  # Temperature controls randomness in the model's output (0.0 to 1.0)
  temperature = 0.7
  
  # Maximum number of tokens to generate in the response
  max-tokens = 1024
  
  # Timeout for API requests in milliseconds
  timeout = 60000
  
  # Whether to enable streaming responses
  enable-streaming = true
  
  # Whether to log API requests
  log-requests = false
  
  # Whether to log API responses
  log-responses = false
}
```

### Environment Variables

Alternatively, you can configure the integration using environment variables:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_MODEL`: The model to use (default: "claude-3-sonnet-20240229")
- `ANTHROPIC_TEMPERATURE`: Temperature parameter (default: 0.7)
- `ANTHROPIC_MAX_TOKENS`: Maximum tokens to generate
- `ANTHROPIC_TIMEOUT_MS`: Timeout in milliseconds (default: 60000)
- `ANTHROPIC_ENABLE_STREAMING`: Whether to enable streaming (default: true)
- `ANTHROPIC_LOG_REQUESTS`: Whether to log requests (default: false)
- `ANTHROPIC_LOG_RESPONSES`: Whether to log responses (default: false)

## Usage

### Basic Example

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.anthropic.AnthropicLLM

val program = for {
  llm <- ZIO.service[LLM]
  completion <- llm.complete("Explain quantum computing in simple terms.")
  _ <- Console.printLine(completion)
} yield ()

// Run with Anthropic integration
program.provide(AnthropicLLM.layer)
```

### Chat Completion Example

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.anthropic.AnthropicLLM

val program = for {
  llm <- ZIO.service[LLM]
  
  messages = Seq(
    ChatMessage.system("You are a helpful assistant that provides concise answers."),
    ChatMessage.user("What are the main features of ZIO?")
  )
  
  response <- llm.completeChat(messages)
  _ <- Console.printLine(response.message.contentAsString)
} yield ()

// Run with Anthropic integration
program.provide(AnthropicLLM.layer)
```

## Advanced Configuration

For more advanced configuration, you can create a custom `AnthropicConfig` and provide it directly:

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.integrations.anthropic.*

val customConfig = AnthropicConfig(
  apiKey = "your-api-key",
  model = "claude-3-opus-20240229",
  temperature = 0.5,
  maxTokens = Some(2000),
  timeout = Duration.fromSeconds(120),
  logResponses = true
)

val customLayer = ZLayer.succeed(customConfig) >>> AnthropicLLM.live

// Use the custom layer
yourProgram.provide(customLayer)
```

## Supported Models

This integration supports all Anthropic Claude models, including:

- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
- claude-2.1
- claude-2.0
- claude-instant-1.2

## Error Handling

All errors from the Anthropic API are properly wrapped in `LLMError` instances, which can be handled using ZIO's error handling mechanisms.