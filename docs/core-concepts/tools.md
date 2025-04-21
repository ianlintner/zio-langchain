---
title: Tools
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Tools

This document explains the Tool abstraction in ZIO LangChain - what tools are, how they work, and how to create and use them in agent-based systems.

## Table of Contents

- [Introduction](#introduction)
- [Core Tool Interface](#core-tool-interface)
- [Built-in Tools](#built-in-tools)
- [Creating Custom Tools](#creating-custom-tools)
- [Tool Integration with Agents](#tool-integration-with-agents)
- [Advanced Tool Concepts](#advanced-tool-concepts)
- [Best Practices](#best-practices)

## Introduction

Tools in ZIO LangChain represent capabilities that agents can use to interact with the world or perform specific functions. They enable agents to:

- Perform calculations and data analysis
- Access external information
- Execute operations on systems
- Interact with APIs and services
- Manipulate files and data
- Generate specialized content

Without tools, agents would be limited to the information contained in their prompts and the knowledge encoded in the LLM. Tools extend agents' capabilities by allowing them to incorporate external functions, systems, and data sources.

## Core Tool Interface

The foundation of tool functionality in ZIO LangChain is the `Tool` trait:

```scala
trait Tool[-R, +E]:
  def name: String
  def description: String
  def execute(input: String): ZIO[R, E, String]
```

Each tool has:
- A **name** that uniquely identifies it to the agent
- A **description** that tells the agent when and how to use the tool
- An **execute** method that performs the actual operation

The description is particularly important, as it serves as documentation for the agent to understand the tool's purpose and required input format.

## Built-in Tools

ZIO LangChain provides several built-in tools for common operations:

### Calculator Tool

```scala
val calculatorTool = Tool.make(
  "calculator", 
  "Calculate mathematical expressions. Input should be a mathematical expression using operators +, -, *, /, ^, (), sqrt, sin, cos, etc."
) { input =>
  ZIO.attempt {
    val expression = input.trim
    val result = /* calculation engine implementation */
    result.toString
  }
}
```

### Search Tool

```scala
val searchTool = Tool.make(
  "search", 
  "Search for information on the internet. Input should be a search query."
) { query =>
  ZIO.attemptBlockingIO {
    // Implementation to search the web
    s"Search results for: $query"
  }
}
```

### Weather Tool

```scala
val weatherTool = Tool.make(
  "weather", 
  "Get current weather for a location. Input should be a city name or postal code."
) { location =>
  ZIO.attemptBlockingIO {
    // Call weather API
    s"The weather in $location is sunny with a temperature of 72°F"
  }
}
```

### Time Tool

```scala
val timeTool = Tool.make(
  "time", 
  "Get the current date and time. No input required, but you can specify a timezone."
) { timezone =>
  ZIO.succeed {
    val now = if (timezone.trim.isEmpty) {
      java.time.LocalDateTime.now()
    } else {
      java.time.LocalDateTime.now(java.time.ZoneId.of(timezone.trim))
    }
    s"Current time: $now"
  }
}
```

### File Tool

```scala
val fileTool = Tool.make(
  "file", 
  "Read the contents of a file. Input should be the file path."
) { path =>
  ZIO.attemptBlockingIO {
    val content = java.nio.file.Files.readString(java.nio.file.Path.of(path.trim))
    content
  }
}
```

## Creating Custom Tools

You can create custom tools to extend agent capabilities for specific use cases:

### Basic Custom Tool

```scala
import zio.*
import zio.langchain.core.tool.*

val databaseTool = Tool.make(
  "database",
  "Query a database. Input should be a SQL query."
) { sql =>
  ZIO.attemptBlockingIO {
    // Execute SQL query
    val resultSet = /* database execution logic */
    // Format results into text
    formatResults(resultSet)
  }
}
```

### Tool with Environment Requirements

```scala
def createApiTool(
  apiKey: String
): Tool[Any, Throwable] = Tool.make(
  "api",
  "Call an external API. Input should be an endpoint name followed by parameters."
) { input =>
  ZIO.attemptBlockingIO {
    val parts = input.split(" ", 2)
    val endpoint = parts(0)
    val params = if (parts.length > 1) parts(1) else ""
    
    // Call API with key
    callApi(endpoint, params, apiKey)
  }
}
```

### Streaming Tool

```scala
def createStreamingTool(): Tool[Any, Throwable] = new Tool[Any, Throwable] {
  override def name: String = "streaming"
  
  override def description: String = 
    "Stream real-time data. Input should be a data source identifier."
  
  override def execute(input: String): ZIO[Any, Throwable, String] =
    ZStream.fromIterator(
      fetchStreamingData(input.trim).iterator
    )
    .take(10) // Limit to first 10 items for agent consumption
    .runCollect
    .map(items => items.mkString("\n"))
}
```

## Tool Integration with Agents

Tools are most powerful when used with agents:

```scala
import zio.*
import zio.langchain.core.agent.*
import zio.langchain.core.tool.*

// Create tools
val tools = Map(
  "calculator" -> calculatorTool,
  "search" -> searchTool,
  "weather" -> weatherTool,
  "time" -> timeTool
)

// Create agent with tools
val agent = ReActAgent(
  llm = llm,
  tools = tools,
  maxIterations = 10
)

// Use the agent with tools
val result = agent.run(
  "What is the square root of 169 multiplied by the current temperature in New York?"
)
```

When the agent receives this query, it might:
1. Use the calculator tool to find the square root of 169 (= 13)
2. Use the weather tool to get the temperature in New York
3. Use the calculator tool again to multiply the results
4. Format a final answer

## Advanced Tool Concepts

### Tool Validation

Implement input validation to prevent errors:

```scala
val validatingTool = Tool.make(
  "validated", 
  "A tool with input validation. Input should be a positive integer."
) { input =>
  ZIO.attempt {
    val number = input.trim.toInt
    if (number <= 0) {
      throw new IllegalArgumentException("Input must be a positive integer")
    }
    s"Processed value: $number"
  }
}
```

### Tool Composition

Combine multiple tools into a single interface:

```scala
def composeTool(
  tools: Map[String, Tool[Any, Throwable]]
): Tool[Any, Throwable] = Tool.make(
  "meta",
  "Access multiple tools. Input should be: 'toolName:toolInput'"
) { input =>
  val parts = input.split(":", 2)
  if (parts.length < 2) {
    ZIO.fail(new IllegalArgumentException("Input must be in format: 'toolName:toolInput'"))
  } else {
    val toolName = parts(0).trim
    val toolInput = parts(1).trim
    
    tools.get(toolName) match {
      case Some(tool) => tool.execute(toolInput)
      case None => ZIO.succeed(s"Tool '$toolName' not found. Available tools: ${tools.keys.mkString(", ")}")
    }
  }
}
```

### Tool Caching

Cache tool results for repeated queries:

```scala
def cachedTool[R, E](
  tool: Tool[R, E],
  cache: Cache[String, Either[E, String]]
): Tool[R, E] = new Tool[R, E] {
  override def name: String = tool.name
  override def description: String = tool.description
  
  override def execute(input: String): ZIO[R, E, String] =
    cache.get(input).flatMap {
      case Some(result) => 
        ZIO.fromEither(result)
      case None =>
        tool.execute(input).tap { result =>
          cache.put(input, Right(result))
        }.catchAll { error =>
          cache.put(input, Left(error)) *> ZIO.fail(error)
        }
    }
}
```

### Tool Metrics

Track tool usage statistics:

```scala
def withMetrics[R, E](
  tool: Tool[R, E],
  metrics: ToolMetrics
): Tool[R, E] = new Tool[R, E] {
  override def name: String = tool.name
  override def description: String = tool.description
  
  override def execute(input: String): ZIO[R, E, String] =
    for
      startTime <- ZIO.succeed(System.currentTimeMillis())
      result <- tool.execute(input)
        .tapBoth(
          error => metrics.recordError(tool.name, error),
          _ => metrics.recordSuccess(tool.name)
        )
      endTime <- ZIO.succeed(System.currentTimeMillis())
      _ <- metrics.recordDuration(tool.name, endTime - startTime)
    yield result
}
```

## Best Practices

1. **Clear Descriptions**: Make tool descriptions specific and clear to help the agent make appropriate decisions:
   ```scala
   // Good description
   "Get weather for a location. Input should be a city name (e.g., 'New York') or postal code."
   
   // Poor description
   "Gets weather data."
   ```

2. **Error Handling**: Implement robust error handling within tools:
   ```scala
   def execute(input: String): ZIO[Any, Throwable, String] =
     ZIO.attempt {
       // Implementation
     }.catchSome {
       case e: NumberFormatException => 
         ZIO.succeed("Invalid number format. Please provide a valid number.")
       case e: IllegalArgumentException =>
         ZIO.succeed(s"Invalid input: ${e.getMessage}")
     }
   ```

3. **Input Parsing**: Parse and validate input carefully:
   ```scala
   val parts = input.split(",").map(_.trim)
   if (parts.length < 2) {
     ZIO.succeed("Error: Input should contain at least two comma-separated values")
   } else {
     // Process valid input
   }
   ```

4. **Output Formatting**: Format outputs consistently for agent consumption:
   ```scala
   // Structured and parsed easily by LLMs
   s"""Results:
      |Temperature: $temperature°F
      |Humidity: $humidity%
      |Wind: $wind mph
      |""".stripMargin
   ```

5. **Rate Limiting**: Implement rate limiting for external API calls:
   ```scala
   val rateLimitedTool = Tool.make(
     "rate-limited-api",
     "API with rate limiting. Input is a query string."
   ) { query =>
     TokenBucket.consume(1) *> apiTool.execute(query)
   }
   ```

6. **Security Considerations**:
   - Validate inputs to prevent injection attacks
   - Implement permission checks for sensitive operations
   - Avoid exposing credentials in tool descriptions or outputs
   - Consider adding confirmation steps for irreversible actions

7. **Testing**: Create comprehensive tests for tools:
   ```scala
   test("Calculator tool handles basic arithmetic") {
     for
       result <- calculatorTool.execute("2 + 2")
     yield assertTrue(result == "4")
   }
   ```

8. **Documentation**: Document tools thoroughly for developers:
   ```scala
   /**
    * Creates a tool for querying weather data.
    *
    * @param apiKey The API key for the weather service
    * @param unitSystem The unit system to use (metric or imperial)
    * @return A tool that can query weather data
    */
   def createWeatherTool(apiKey: String, unitSystem: String): Tool[Any, Throwable]