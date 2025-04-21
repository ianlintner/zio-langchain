---
title: Simple RAG Example
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Simple RAG Example

This example demonstrates how to implement a basic Retrieval-Augmented Generation (RAG) system using ZIO LangChain. RAG enhances LLM responses by retrieving relevant information from a document collection.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Implementation](#implementation)
- [Step-by-Step Explanation](#step-by-step-explanation)
- [Running the Example](#running-the-example)
- [Extending the Example](#extending-the-example)
- [Common Issues](#common-issues)

## Overview

The Simple RAG example shows how to:

1. Load and process documents
2. Create vector embeddings for documents
3. Implement a similarity-based retrieval system
4. Generate responses using retrieved context

This pattern is useful for question-answering systems, chatbots with specific knowledge, and other applications where you want to ground LLM responses in factual information.

## Prerequisites

Before running this example, ensure you have:

- Set up ZIO LangChain (see [Installation](../getting-started/installation.md))
- Configured your OpenAI API key (see [Configuration](../getting-started/configuration.md))
- Basic understanding of embeddings (see [Embeddings](../core-concepts/embeddings.md))

## Implementation

Here's the complete implementation of a simple RAG system:

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

## Step-by-Step Explanation

Let's break down how this example works:

### 1. Document Preparation

```scala
val documents = Seq(
  Document(
    id = "doc1",
    content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
    metadata = Map("source" -> "docs")
  ),
  // ...more documents
)
```

We start with a collection of documents that contain information we want to query. In a real application, these would typically be loaded from files, a database, or an API.

### 2. Creating Embeddings

```scala
embeddedDocs <- embeddingModel.embedDocuments(documents)
  .mapError(e => new RuntimeException(s"Embedding error: ${e.message}", e))
```

We convert each document into a vector embedding using an embedding model (OpenAI's in this case). These embeddings capture the semantic meaning of each document's content.

### 3. Implementing a Retriever

```scala
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
```

The retriever:
1. Creates an embedding for the user's query
2. Calculates the cosine similarity between the query embedding and each document embedding
3. Returns the top N most similar documents

### 4. Query Processing

```scala
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
```

When a user enters a query:
1. We retrieve the most relevant documents
2. Format them into a context string
3. Create a prompt that includes both the context and the user's query
4. Send this to the LLM to generate a response

## Running the Example

You can run this example using the provided script:

```bash
export OPENAI_API_KEY=your-api-key
./run-examples.sh SimpleRAG
```

Or directly with SBT:

```bash
export OPENAI_API_KEY=your-api-key
sbt "examples/runMain zio.langchain.examples.SimpleRAG"
```

## Extending the Example

There are several ways to enhance this simple RAG implementation:

1. **Document Loading**: Replace the hardcoded documents with actual document loading
   ```scala
   val documentLoader = new TextFileLoader(Path.of("docs/"))
   val documents = documentLoader.load.runCollect.map(_.toSeq)
   ```

2. **Document Chunking**: Split larger documents into smaller chunks
   ```scala
   val documentParser = new DocumentParser(maxChunkSize = 1000, overlap = 200)
   val chunks = documents.flatMap(documentParser.splitDocument)
   ```

3. **Vector Store Integration**: Use a proper vector database instead of in-memory storage
   ```scala
   val vectorStore = new ChromaVectorStore(
     embeddings = embeddingModel,
     collectionName = "documentation"
   )
   ```

4. **Better Prompt Engineering**: Improve the prompt template
   ```scala
   val promptTemplate = RetrievalPromptTemplate(
     template = """You are a helpful assistant for ZIO.
                 |
                 |Context information is below.
                 |---------------------
                 |{context}
                 |---------------------
                 |
                 |Given the context information and not prior knowledge, answer the query.
                 |Query: {query}
                 |Answer:""".stripMargin,
     inputVariables = Set("context", "query")
   )
   ```

## Common Issues

- **Token Limits**: If your documents or context is too large, you may exceed token limits
- **Embedding Quality**: The quality of your embeddings affects retrieval performance
- **Document Relevance**: Ensure your document collection contains relevant information
- **Query Formulation**: How a query is phrased can affect which documents are retrieved