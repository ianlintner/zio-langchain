---
title: Quick Start Guide
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Quick Start Guide

This guide will help you get up and running with ZIO LangChain quickly, demonstrating basic usage patterns with practical examples.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Basic Chat Example](#basic-chat-example)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [RAG Example](#rag-example)
- [Common Patterns](#common-patterns)
- [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have:

1. Added ZIO LangChain dependencies to your project (see [Installation](installation.md))
2. Obtained API keys for your LLM provider (see [Configuration](configuration.md))
3. Set up a basic Scala project with SBT

## Basic Chat Example

Here's a complete example of a simple chat application using ZIO LangChain with OpenAI:

```scala
import zio.*
import zio.Console.*
import zio.langchain.core.model.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object SimpleChat extends ZIOAppDefault {
  val program = for {
    // Get LLM service
    llm <- ZIO.service[LLM]
    
    // Welcome message
    _ <- printLine("Welcome to ZIO LangChain Chat!")
    _ <- printLine("Type 'exit' to quit.")
    
    // Chat loop
    _ <- (for {
      // Get user input
      _ <- printLine("\nYou: ")
      input <- readLine
      
      // Check for exit condition
      _ <- ZIO.when(input.trim.toLowerCase != "exit") {
        for {
          // Get LLM response
          response <- llm.complete(input)
          
          // Display response
          _ <- printLine(s"\nAI: $response")
        } yield ()
      }
    } yield input).repeatWhile(input => input.trim.toLowerCase != "exit")
    
    // Goodbye message
    _ <- printLine("\nThank you for using ZIO LangChain Chat!")
  } yield ()
  
  override def run = program.provide(
    // Provide LLM implementation
    OpenAILLM.live,
    // Provide configuration
    ZLayer.succeed(
      OpenAIConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = "gpt-3.5-turbo"
      )
    )
  )
}
```

Run this example with:

```bash
export OPENAI_API_KEY=your-api-key
sbt run
```

## Step-by-Step Walkthrough

Let's break down the example above:

### 1. Import Dependencies

```scala
import zio.*
import zio.Console.*
import zio.langchain.core.model.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*
```

These imports provide access to ZIO, Console utilities, and ZIO LangChain components.

### 2. Create ZIO Program

```scala
val program = for {
  // Get LLM service
  llm <- ZIO.service[LLM]
  
  // Welcome message
  _ <- printLine("Welcome to ZIO LangChain Chat!")
  _ <- printLine("Type 'exit' to quit.")
  
  // Chat loop
  _ <- (for {
    // Get user input
    _ <- printLine("\nYou: ")
    input <- readLine
    
    // Check for exit condition
    _ <- ZIO.when(input.trim.toLowerCase != "exit") {
      for {
        // Get LLM response
        response <- llm.complete(input)
        
        // Display response
        _ <- printLine(s"\nAI: $response")
      } yield ()
    }
  } yield input).repeatWhile(input => input.trim.toLowerCase != "exit")
  
  // Goodbye message
  _ <- printLine("\nThank you for using ZIO LangChain Chat!")
} yield ()
```

This defines a ZIO effect that:
- Obtains the LLM service from the environment
- Displays welcome messages
- Enters a chat loop that continues until the user types "exit"
- For each user input, sends it to the LLM and displays the response

### 3. Provide Dependencies

```scala
override def run = program.provide(
  // Provide LLM implementation
  OpenAILLM.live,
  // Provide configuration
  ZLayer.succeed(
    OpenAIConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "gpt-3.5-turbo"
    )
  )
)
```

This provides the necessary dependencies to run the program:
- `OpenAILLM.live`: The live implementation of the LLM service
- Configuration for OpenAI with API key and model

## RAG Example

Here's a quick example of Retrieval-Augmented Generation (RAG):

```scala
import zio.*
import zio.Console.*
import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object SimpleRAG extends ZIOAppDefault {
  // Sample documents
  val documents = Seq(
    Document(
      id = "doc1",
      content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
      metadata = Map("source" -> "docs")
    ),
    Document(
      id = "doc2",
      content = "ZIO provides a powerful effect system for functional programming.",
      metadata = Map("source" -> "docs")
    ),
    Document(
      id = "doc3",
      content = "ZIO LangChain is a Scala wrapper around langchain4j with ZIO integration.",
      metadata = Map("source" -> "docs")
    )
  )
  
  // Simple in-memory retriever
  def createRetriever(
    embeddedDocs: Seq[(Document, Embedding)]
  ): Retriever = new Retriever {
    override def retrieve(
      query: String, 
      maxResults: Int = 2
    ): ZIO[Any, RetrieverError, Seq[Document]] =
      for {
        embeddingModel <- ZIO.service[EmbeddingModel]
        queryEmbedding <- embeddingModel.embed(query)
          .mapError(e => RetrieverError(e))
          
        // Find most similar documents using cosine similarity
        similarities = embeddedDocs.map { case (doc, embedding) =>
          (doc, queryEmbedding.cosineSimilarity(embedding))
        }
        
        // Sort by similarity (highest first) and take top results
        topDocs = similarities.sortBy(-_._2).take(maxResults).map(_._1)
      } yield topDocs
  }
  
  val program = for {
    // Get services
    embeddingModel <- ZIO.service[EmbeddingModel]
    llm <- ZIO.service[LLM]
    
    // Create document embeddings
    embeddedDocs <- embeddingModel.embedDocuments(documents)
      .mapError(e => new RuntimeException(s"Embedding error: ${e.message}", e))
    
    // Create retriever
    retriever = createRetriever(embeddedDocs)
    
    // Welcome message
    _ <- printLine("RAG System: Ask a question about ZIO")
    _ <- printLine("Type 'exit' to quit")
    
    // Query loop
    _ <- (for {
      // Get user query
      _ <- printLine("\nQuestion: ")
      query <- readLine
      
      // Check for exit condition
      _ <- ZIO.when(query.trim.toLowerCase != "exit") {
        for {
          // Retrieve relevant documents
          relevantDocs <- retriever.retrieve(query)
          
          // Format context for the prompt
          context = relevantDocs.map(_.content).mkString("\n\n")
          
          // Create prompt with context
          prompt = s"""Based on the following information:
                      |
                      |$context
                      |
                      |Question: $query
                      |
                      |Answer:""".stripMargin
          
          // Get response from LLM
          response <- llm.complete(prompt)
          
          // Display response
          _ <- printLine(s"\nAnswer: $response")
        } yield ()
      }
    } yield query).repeatWhile(query => query.trim.toLowerCase != "exit")
    
    // Goodbye message
    _ <- printLine("\nThank you for using the RAG system!")
  } yield ()
  
  override def run = program.provide(
    OpenAILLM.live,
    OpenAIEmbedding.live,
    ZLayer.succeed(
      OpenAIConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = "gpt-3.5-turbo"
      )
    ),
    ZLayer.succeed(
      OpenAIEmbeddingConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = "text-embedding-ada-002"
      )
    )
  )
}
```

## Common Patterns

### Chat with History

```scala
import zio.langchain.core.memory.*

// Create a memory component
val memory = new VolatileMemory()

// Add system message
memory.add(ChatMessage(Role.System, "You are a helpful assistant."))

// Add user message and get response
for {
  _ <- memory.add(ChatMessage(Role.User, userInput))
  history <- memory.get
  response <- llm.completeChat(history)
  _ <- memory.add(response.message)
} yield response.message.content
```

### Streaming Responses

```scala
import zio.stream.*

// Configure for streaming
val streamingConfig = OpenAIConfig(
  apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
  model = "gpt-3.5-turbo",
  streaming = true
)

// Get streaming response
for {
  llm <- ZIO.service[StreamingLLM]
  stream <- llm.completeStreaming(prompt)
  _ <- stream.foreach { chunk =>
    Console.print(chunk).orDie
  }
} yield ()
```

### Simple Agent

```scala
import zio.langchain.core.agent.*
import zio.langchain.core.tool.*

// Create tools
val calculatorTool = Tool.make(
  "calculator", 
  "Calculate mathematical expressions"
) { input =>
  ZIO.attempt {
    val expr = input.trim
    val result = /* calculation logic */
    result.toString
  }
}

// Create agent
val agent = ReActAgent(
  llm = llm,
  tools = Map("calculator" -> calculatorTool),
  maxIterations = 5
)

// Use agent
agent.run("What is the square root of 144 plus 10?")
```

## Next Steps

Now that you've seen the basics, you can:

1. Explore [Core Concepts](../core-concepts/index.md) for deeper understanding
2. Check out more [Examples](../examples/index.md) for inspiration
3. Learn about specific components:
   - [LLM Integration](../core-concepts/llm-integration.md)
   - [Chains](../core-concepts/chains.md)
   - [Agents](../core-concepts/agents.md)
   - [Retrievers](../core-concepts/retrieval.md)

For a complete application, see the [Simple RAG Example](../examples/simple-rag.md) in the examples section.