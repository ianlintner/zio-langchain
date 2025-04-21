---
title: Quick Start Guide
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Quick Start Guide

This guide will help you quickly build your first ZIO LangChain application. We'll create a simple chat application that interacts with an LLM.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Simple Chat Example](#simple-chat-example)
- [Running the Example](#running-the-example)
- [Building a Retrieval-Augmented Generation (RAG) System](#building-a-retrieval-augmented-generation-rag-system)
- [Next Steps](#next-steps)

## Prerequisites

Before starting, ensure you have:

1. [Installed](installation.md) ZIO LangChain
2. [Configured](configuration.md) your API keys

## Project Setup

Create a new Scala project with SBT:

```bash
sbt new scala/scala3.g8
```

Then add the required dependencies to your `build.sbt` file:

```scala
val zioVersion = "2.0.19"
val zioLangchainVersion = "0.1.0-SNAPSHOT"

libraryDependencies ++= Seq(
  "dev.zio" %% "zio" % zioVersion,
  "dev.zio" %% "zio-langchain-core" % zioLangchainVersion,
  "dev.zio" %% "zio-langchain-openai" % zioLangchainVersion
)
```

## Simple Chat Example

Let's create a simple chat application that interacts with an LLM. Create a file named `SimpleChat.scala`:

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object SimpleChat extends ZIOAppDefault:
  override def run =
    for
      // Get the LLM service
      llm <- ZIO.service[LLM]
      
      // Prompt for user input
      _ <- Console.printLine("Enter your message (or 'exit' to quit):")
      
      // Chat loop
      _ <- (for
        userInput <- Console.readLine
        _ <- ZIO.when(userInput.trim.toLowerCase != "exit") {
          for
            // Send the user input to the LLM
            response <- llm.complete(userInput)
            
            // Display the response
            _ <- Console.printLine(s"AI: $response")
            
            // Prompt for next input
            _ <- Console.printLine("\nEnter your message (or 'exit' to quit):")
          yield ()
        }
      ).repeatWhile(input => input.trim.toLowerCase != "exit")
    yield ()
    .provide(
      // Provide the LLM implementation
      OpenAILLM.live,
      
      // Provide the configuration
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo"
        )
      )
    )
```

This simple application:

1. Creates a ZIO application using `ZIOAppDefault`
2. Sets up a chat loop that:
   - Prompts the user for input
   - Sends the input to the LLM
   - Displays the response
   - Repeats until the user types "exit"
3. Provides the necessary dependencies:
   - OpenAI LLM implementation
   - Configuration with API key from environment variables

## Running the Example

Run the example with:

```bash
export OPENAI_API_KEY=your-api-key
sbt run
```

You should see a prompt where you can enter messages and receive responses from the AI.

## Building a Retrieval-Augmented Generation (RAG) System

Let's build a more advanced example that demonstrates Retrieval-Augmented Generation. Create a file named `SimpleRAG.scala`:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

object SimpleRAG extends ZIOAppDefault:
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

  // Create a simple in-memory retriever
  def createRetriever(
    embeddedDocs: Seq[(Document, Embedding)]
  ): Retriever = new Retriever:
    override def retrieve(
      query: String, 
      maxResults: Int = 2
    ): ZIO[Any, RetrieverError, Seq[Document]] =
      for
        embeddingModel <- ZIO.service[EmbeddingModel]
        queryEmbedding <- embeddingModel.embed(query)
          .mapError(e => RetrieverError(e))
          
        // Find most similar documents using cosine similarity
        similarities = embeddedDocs.map { case (doc, embedding) =>
          (doc, queryEmbedding.cosineSimilarity(embedding))
        }
        
        // Sort by similarity (highest first) and take top results
        topDocs = similarities.sortBy(-_._2).take(maxResults).map(_._1)
      yield topDocs

  override def run =
    for
      // Get the embedding model
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Get the LLM
      llm <- ZIO.service[LLM]
      
      // Create embeddings for documents
      embeddedDocs <- embeddingModel.embedDocuments(documents)
        .mapError(e => new RuntimeException(s"Embedding error: ${e.message}", e))
      
      // Create retriever
      retriever = createRetriever(embeddedDocs)
      
      // Chat loop
      _ <- Console.printLine("Enter your question about ZIO (or 'exit' to quit):")
      _ <- (for
        userInput <- Console.readLine
        _ <- ZIO.when(userInput.trim.toLowerCase != "exit") {
          for
            // Retrieve relevant documents
            relevantDocs <- retriever.retrieve(userInput)
            
            // Format context for the prompt
            context = relevantDocs.map(_.content).mkString("\n\n")
            
            // Create prompt with context
            prompt = s"""Based on the following information:
                      |
                      |$context
                      |
                      |Question: $userInput
                      |
                      |Answer:""".stripMargin
            
            // Get response from LLM
            response <- llm.complete(prompt)
            
            // Display response
            _ <- Console.printLine(s"AI: $response")
            _ <- Console.printLine("\nEnter your question about ZIO (or 'exit' to quit):")
          yield ()
        }
      ).repeatWhile(input => input.trim.toLowerCase != "exit")
    yield ()
    .provide(
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
```

This RAG example:

1. Defines sample documents (in a real app, you'd load these from files)
2. Creates embeddings for the documents
3. Implements a simple retriever that finds similar documents using cosine similarity
4. Retrieves relevant documents based on the user's query
5. Creates a prompt with the retrieved context
6. Sends the prompt to the LLM to generate a response

## Next Steps

Now that you've built your first ZIO LangChain applications, you can:

1. Explore more [Core Concepts](../core-concepts/llm-integration.md)
2. Learn about different [Components](../components/models/index.md)
3. Check the [Examples](../examples/index.md) for more advanced use cases
4. Review the [API Documentation](../api/index.md) for detailed reference

For more complex applications, see the examples directory in the project repository, which contains implementations of chat systems, RAG applications, and agent-based systems.