# ZIO LangChain RAG Module

The RAG (Retrieval-Augmented Generation) module provides advanced techniques for improving retrieval performance in RAG systems. It focuses on query transformation strategies that enhance the quality of search queries before they are used for document retrieval.

## Overview

RAG systems combine retrieval of relevant documents with generative AI to produce more accurate, factual, and contextually relevant responses. The quality of retrieval significantly impacts the overall performance of RAG systems. This module provides tools to improve retrieval through query transformation techniques.

## Features

- **Query Transformation Interface**: A flexible interface for implementing various query transformation strategies
- **LLM-Powered Query Expansion**: Expands queries with relevant terms and context
- **Hypothetical Document Embeddings (HyDE)**: Generates a hypothetical answer to use as the query
- **Multi-Query Transformation**: Generates multiple variations of a query to capture different aspects
- **Composable Transformers**: Chain multiple transformations together for enhanced performance
- **Transforming Retrievers**: Wrap existing retrievers with query transformation capabilities
- **ZIO Layer Integration**: Full integration with ZIO's layer system for dependency injection

## Usage

### Basic Query Transformation

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.rag.*

// Create a query expansion transformer
val program = for {
  transformer <- ZIO.service[QueryTransformer]
  
  // Transform a query
  originalQuery = "How does ZIO work?"
  transformedQuery <- transformer.transform(originalQuery)
  
  _ <- ZIO.logInfo(s"Original query: $originalQuery")
  _ <- ZIO.logInfo(s"Transformed query: $transformedQuery")
} yield ()

// Provide the transformer layer
program.provide(
  queryExpansionTransformer,
  // Requires an LLM implementation
  myLLMLayer
)
```

### Enhancing a Retriever with Query Transformation

```scala
import zio.*
import zio.langchain.core.retriever.Retriever
import zio.langchain.core.model.LLM
import zio.langchain.rag.*

val program = for {
  // Get the services
  retriever <- ZIO.service[Retriever]
  
  // Use the retriever
  query = "What is functional programming?"
  documents <- retriever.retrieve(query, maxResults = 5)
  
  _ <- ZIO.logInfo(s"Retrieved ${documents.size} documents for query: $query")
} yield ()

// Provide the layers
program.provide(
  // This will use the transformer before retrieval
  transformingRetriever,
  // The underlying retriever
  myRetrieverLayer,
  // The query transformer
  hydeTransformer,
  // Required by the transformer
  myLLMLayer
)
```

### Using Multiple Query Transformations

```scala
import zio.*
import zio.langchain.core.retriever.Retriever
import zio.langchain.core.model.LLM
import zio.langchain.rag.*

// Create a retriever that uses multiple transformations
val myMultiTransformingRetriever = for {
  baseRetriever <- ZIO.service[Retriever]
  queryExpansion <- ZIO.service[QueryTransformer].provide(queryExpansionTransformer)
  hyde <- ZIO.service[QueryTransformer].provide(hydeTransformer)
  
  // Create a multi-transforming retriever
  retriever = createMultiTransformingRetriever(
    baseRetriever,
    Seq(queryExpansion, hyde)
  )
} yield retriever

// Use the retriever
val program = for {
  retriever <- myMultiTransformingRetriever
  documents <- retriever.retrieve("How does ZIO handle concurrency?", 5)
  _ <- ZIO.logInfo(s"Retrieved ${documents.size} documents")
} yield ()

// Provide the required layers
program.provide(
  myRetrieverLayer,
  myLLMLayer
)
```

## Transformation Strategies

### Query Expansion

Expands the original query to include more relevant terms, synonyms, or context. This helps bridge the vocabulary gap between user queries and document content.

### Hypothetical Document Embeddings (HyDE)

Generates a hypothetical document that would answer the query, then uses that document as the query for retrieval. This leverages the LLM's knowledge to create a more comprehensive search query.

### Multi-Query Transformation

Generates multiple variations of the original query to capture different aspects or interpretations. Results from all variations are combined to provide more comprehensive retrieval.

## Integration with Existing Retrievers

The module is designed to work seamlessly with existing retrievers:

```scala
import zio.langchain.rag.*

// Wrap an existing retriever with query transformation
val enhancedRetriever = TransformingRetriever(
  existingRetriever,
  queryTransformer
)

// Use multiple transformations
val multiRetriever = MultiTransformingRetriever(
  existingRetriever,
  Seq(transformer1, transformer2, transformer3)
)
```

## Composing Transformers

Transformers can be composed to apply multiple transformations in sequence:

```scala
import zio.langchain.rag.*

// Compose transformers
val compositeTransformer = composeTransformers(
  queryExpansionTransformer,
  hydeTransformer
)
```

## Example

See `modules/examples/src/main/scala/zio/langchain/examples/QueryTransformationExample.scala` for a complete example of using query transformation techniques.