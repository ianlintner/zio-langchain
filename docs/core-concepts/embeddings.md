---
title: Embeddings
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Embeddings

This document explains embeddings in ZIO LangChain - what they are, how they work, and how to use them effectively for semantic search and retrieval.

## Table of Contents

- [Introduction](#introduction)
- [Core Embedding Interface](#core-embedding-interface)
- [Available Implementations](#available-implementations)
- [Working with Embeddings](#working-with-embeddings)
- [Similarity Measures](#similarity-measures)
- [Vector Storage](#vector-storage)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Introduction

Embeddings are numerical representations of text that capture semantic meaning in a vector space. They enable:

- **Semantic search**: Find documents based on meaning, not just keywords
- **Clustering**: Group similar texts together
- **Visualization**: Map relationships between concepts
- **Recommendations**: Suggest related content

In ZIO LangChain, embeddings are a foundational technology that powers many higher-level features, particularly retrieval systems used in RAG (Retrieval-Augmented Generation) applications.

Unlike traditional keyword search, which relies on exact word matches, embedding-based search can find relevant documents even when they share few or no exact keywords with the query.

## Core Embedding Interface

The foundation of embedding functionality in ZIO LangChain is the `EmbeddingModel` trait:

```scala
trait EmbeddingModel:
  def embed(text: String): ZIO[Any, EmbeddingError, Embedding]
  
  def embedDocuments(
    documents: Seq[Document]
  ): ZIO[Any, EmbeddingError, Seq[(Document, Embedding)]]
```

This interface provides:

- **embed**: Convert a single text string into an embedding vector
- **embedDocuments**: Convert multiple documents into embedding vectors, maintaining the association between documents and their embeddings

All operations return `ZIO` effects with an error channel for proper error handling.

The `Embedding` class represents the vector output:

```scala
case class Embedding(vector: Seq[Double]) {
  def cosineSimilarity(other: Embedding): Double = {
    // Implementation of cosine similarity calculation
  }
  
  def dotProduct(other: Embedding): Double = {
    // Implementation of dot product calculation
  }
  
  def euclideanDistance(other: Embedding): Double = {
    // Implementation of Euclidean distance calculation
  }
}
```

This class not only stores the vector but also provides similarity calculation methods to compare embeddings.

## Available Implementations

### OpenAI Embeddings

The most widely used embedding model implementation in ZIO LangChain is the OpenAI embedding service:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.integrations.openai.*

val program = for {
  embeddingModel <- ZIO.service[EmbeddingModel]
  embedding <- embeddingModel.embed("ZIO is a Scala library for asynchronous programming.")
  _ <- Console.printLine(s"Embedding dimension: ${embedding.vector.size}")
} yield ()

program.provide(
  OpenAIEmbedding.live,
  ZLayer.succeed(
    OpenAIEmbeddingConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "text-embedding-ada-002"
    )
  )
)
```

OpenAI's `text-embedding-ada-002` model produces 1536-dimensional vectors that capture semantic relationships between texts.

### HuggingFace Embeddings

For open-source alternatives, ZIO LangChain supports HuggingFace embedding models:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.integrations.huggingface.*

val program = for {
  embeddingModel <- ZIO.service[EmbeddingModel]
  embedding <- embeddingModel.embed("ZIO is a Scala library for asynchronous programming.")
  _ <- Console.printLine(s"Embedding dimension: ${embedding.vector.size}")
} yield ()

program.provide(
  HuggingFaceEmbedding.live,
  ZLayer.succeed(
    HuggingFaceEmbeddingConfig(
      apiKey = sys.env.getOrElse("HF_API_KEY", ""),
      model = "sentence-transformers/all-MiniLM-L6-v2"
    )
  )
)
```

The HuggingFace integration supports various models with different characteristics:

| Model | Dimensions | Features |
|-------|------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Compact, fast, good for general text similarity |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality, but larger and slower |
| `BAAI/bge-small-en` | 384 | Optimized for English retrieval tasks |

### Custom Embeddings

You can also implement custom embedding models:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.domain.*

class TfIdfEmbedding(corpus: Seq[String]) extends EmbeddingModel:
  // Simple TF-IDF implementation
  private val (vocabulary, idfValues) = computeVocabularyAndIdf(corpus)
  
  override def embed(text: String): ZIO[Any, EmbeddingError, Embedding] =
    ZIO.attempt {
      val vector = computeTfIdfVector(text, vocabulary, idfValues)
      Embedding(vector)
    }.mapError(e => EmbeddingError(e))
  
  override def embedDocuments(documents: Seq[Document]): ZIO[Any, EmbeddingError, Seq[(Document, Embedding)]] =
    ZIO.foreach(documents)(doc => 
      embed(doc.content).map(embedding => (doc, embedding))
    )
  
  // Helper methods for TF-IDF computation
  private def computeVocabularyAndIdf(corpus: Seq[String]): (Map[String, Int], Map[String, Double]) = {
    // Implementation details
    (Map.empty, Map.empty) // Placeholder
  }
  
  private def computeTfIdfVector(text: String, vocabulary: Map[String, Int], idfValues: Map[String, Double]): Seq[Double] = {
    // Implementation details
    Seq.empty // Placeholder
  }
```

## Working with Embeddings

### Creating Embeddings

The basic operation is to convert text into embeddings:

```scala
// Single text embedding
val textEmbedding = embeddingModel.embed("This is a sample text.")

// Multiple document embeddings
val documents = Seq(
  Document("doc1", "ZIO is a library for asynchronous programming."),
  Document("doc2", "Scala is a functional programming language.")
)
val documentEmbeddings = embeddingModel.embedDocuments(documents)
```

### Comparing Embeddings

Once you have embeddings, you can compare them using similarity measures:

```scala
val similarityProgram = for {
  embeddingModel <- ZIO.service[EmbeddingModel]
  
  // Create embeddings
  embedding1 <- embeddingModel.embed("ZIO provides asynchronous, concurrent programming in Scala")
  embedding2 <- embeddingModel.embed("ZIO offers concurrency primitives for async Scala applications")
  embedding3 <- embeddingModel.embed("Python is a widely used programming language")
  
  // Calculate similarities
  similarity12 = embedding1.cosineSimilarity(embedding2)
  similarity13 = embedding1.cosineSimilarity(embedding3)
  
  // Print results
  _ <- Console.printLine(s"Similarity between related texts: $similarity12")
  _ <- Console.printLine(s"Similarity between unrelated texts: $similarity13")
} yield ()
```

Typically, related texts will have higher similarity scores than unrelated ones.

## Similarity Measures

ZIO LangChain supports multiple similarity measures for comparing embeddings:

### Cosine Similarity

The most commonly used similarity measure, which compares the angle between vectors regardless of magnitude:

```scala
def cosineSimilarity(a: Embedding, b: Embedding): Double = {
  val dotProduct = a.vector.zip(b.vector).map { case (x, y) => x * y }.sum
  val normA = math.sqrt(a.vector.map(x => x * x).sum)
  val normB = math.sqrt(b.vector.map(x => x * x).sum)
  dotProduct / (normA * normB)
}
```

Cosine similarity ranges from -1 (opposite) to 1 (identical), with 0 indicating orthogonality (no relationship).

### Dot Product

A simpler measure that works well for normalized vectors:

```scala
def dotProduct(a: Embedding, b: Embedding): Double = {
  a.vector.zip(b.vector).map { case (x, y) => x * y }.sum
}
```

### Euclidean Distance

Measures the straight-line distance between vectors, smaller values indicate more similarity:

```scala
def euclideanDistance(a: Embedding, b: Embedding): Double = {
  math.sqrt(
    a.vector.zip(b.vector).map { case (x, y) => math.pow(x - y, 2) }.sum
  )
}
```

## Vector Storage

For practical applications, embeddings are typically stored in a vector database or vector store that supports efficient similarity search.

### In-Memory Storage

For small applications, a simple in-memory solution can work:

```scala
class InMemoryVectorStore(
  embeddingModel: EmbeddingModel
) {
  private val store = scala.collection.mutable.Map.empty[String, (Document, Embedding)]
  
  def addDocument(document: Document): ZIO[Any, EmbeddingError, Unit] =
    for {
      embedding <- embeddingModel.embed(document.content)
      _ <- ZIO.succeed(store.put(document.id, (document, embedding)))
    } yield ()
    
  def searchSimilar(
    query: String, 
    maxResults: Int = 5
  ): ZIO[Any, EmbeddingError, Seq[(Document, Double)]] =
    for {
      queryEmbedding <- embeddingModel.embed(query)
      
      // Calculate similarities
      similarities = store.values.map { case (doc, embedding) =>
        (doc, queryEmbedding.cosineSimilarity(embedding))
      }
      
      // Sort by similarity (descending) and take top results
      results = similarities.toSeq.sortBy(-_._2).take(maxResults)
    } yield results
}
```

### External Vector Databases

For production use, consider integrating with specialized vector databases:

- **Chroma**: Open-source embedding database
- **Pinecone**: Managed vector database service
- **Weaviate**: Vector search engine
- **Milvus**: Open-source vector database
- **FAISS**: Facebook AI Similarity Search library

## Usage Examples

### Simple RAG System

Here's a basic RAG (Retrieval-Augmented Generation) implementation:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.domain.*

def createRagSystem(
  documents: Seq[Document],
  embeddingModel: EmbeddingModel,
  llm: LLM
): ZIO[Any, Throwable, String => Task[String]] = {
  // Create document embeddings
  val embeddedDocsEffect = embeddingModel.embedDocuments(documents)
    .mapError(e => new RuntimeException(s"Embedding error: ${e.message}", e))
  
  // Create retriever
  embeddedDocsEffect.map { embeddedDocs =>
    // Create closure that performs RAG
    (query: String) =>
      for {
        // Create query embedding
        queryEmbedding <- embeddingModel.embed(query)
          .mapError(e => new RuntimeException(s"Embedding error: ${e.message}", e))
        
        // Find similar documents
        similarities = embeddedDocs.map { case (doc, embedding) =>
          (doc, queryEmbedding.cosineSimilarity(embedding))
        }
        
        relevantDocs = similarities.sortBy(-_._2).take(3).map(_._1)
        
        // Create context from relevant documents
        context = relevantDocs.map(_.content).mkString("\n\n")
        
        // Create prompt with context
        prompt = s"""Answer the question based on the following context:
                   |
                   |$context
                   |
                   |Question: $query
                   |Answer:""".stripMargin
        
        // Get LLM response
        response <- llm.complete(prompt)
      } yield response
  }
}

// Usage:
val program = for {
  embeddingModel <- ZIO.service[EmbeddingModel]
  llm <- ZIO.service[LLM]
  
  // Sample documents
  documents = Seq(
    Document("1", "ZIO is a library for asynchronous and concurrent programming in Scala."),
    Document("2", "ZIO provides composable ZIO data types for managing resources."),
    Document("3", "Effects in ZIO represent programs that can fail, succeed, or need environment.")
  )
  
  // Create RAG system
  ragSystem <- createRagSystem(documents, embeddingModel, llm)
  
  // Use the system
  response <- ragSystem("How does ZIO handle errors?")
  
  // Print response
  _ <- Console.printLine(response)
} yield ()
```

### Semantic Search

Create a simple semantic search engine:

```scala
import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.domain.*

class SemanticSearchEngine(
  documents: Seq[Document],
  embeddingModel: EmbeddingModel
) {
  // Initialize embeddings
  private val embeddedDocsEffect = embeddingModel.embedDocuments(documents)
  
  // Search function
  def search(
    query: String, 
    maxResults: Int = 5
  ): ZIO[Any, EmbeddingError, Seq[(Document, Double)]] =
    for {
      // Get embedded documents
      embeddedDocs <- embeddedDocsEffect
      
      // Create query embedding
      queryEmbedding <- embeddingModel.embed(query)
      
      // Calculate similarities
      similarities = embeddedDocs.map { case (doc, embedding) =>
        (doc, queryEmbedding.cosineSimilarity(embedding))
      }
      
      // Sort by similarity and take top results
      results = similarities.sortBy(-_._2).take(maxResults)
    } yield results
}
```

## Best Practices

### Embedding Production

1. **Chunk Documents Appropriately**:
   - Too large: May dilute meaning
   - Too small: May lose context
   - Ideal size depends on your use case (typically 256-1024 tokens)

2. **Process Text Before Embedding**:
   ```scala
   def preprocessText(text: String): String = {
     text.trim
       .replaceAll("\\s+", " ")
       .toLowerCase
   }
   ```

3. **Handle Long Texts**:
   - Split into smaller chunks with overlap
   - Consider hierarchical embedding approaches
   ```scala
   def chunkText(text: String, chunkSize: Int, overlap: Int): Seq[String] = {
     // Implementation that splits text into chunks with overlap
   }
   ```

4. **Batch Embedding Requests**:
   - Process documents in batches to reduce API calls
   ```scala
   val batchSize = 16
   documents.grouped(batchSize).flatMap { batch =>
     embeddingModel.embedDocuments(batch).runCollect.map(_.toSeq)
   }
   ```

### Embedding Retrieval

1. **Use Approximate Nearest Neighbor Search** for large collections:
   - Exact search is O(n) and doesn't scale
   - ANN algorithms trade perfect accuracy for orders of magnitude speed improvements
   - Consider libraries like HNSW, IVF, or Product Quantization

2. **Filter Before or After Embedding**:
   - Pre-filtering: Filter documents before embedding (more efficient)
   ```scala
   val filteredDocs = documents.filter(doc => doc.metadata.get("date") == Some("2023"))
   val embeddedFilteredDocs = embeddingModel.embedDocuments(filteredDocs)
   ```
   
   - Post-filtering: Retrieve semantically similar docs then filter (more flexible)
   ```scala
   val results = similarities.filter { case (doc, _) => 
     doc.metadata.get("category") == Some("technology") 
   }.sortBy(-_._2).take(maxResults)
   ```

3. **Combine with Keyword Search**:
   - Hybrid search often outperforms pure semantic or keyword approaches
   ```scala
   def hybridSearch(
     query: String, 
     documents: Seq[Document],
     embeddingModel: EmbeddingModel,
     keywordWeight: Double = 0.3,
     semanticWeight: Double = 0.7
   ): ZIO[Any, EmbeddingError, Seq[(Document, Double)]] = {
     // Implementation that combines keyword and semantic search
   }
   ```

4. **Store Metadata With Embeddings**:
   - Keep metadata alongside vectors for filtering
   - Common metadata: source, author, date, category, etc.

5. **Normalize Vectors** if using dot product similarity measure:
   ```scala
   def normalizeVector(vector: Seq[Double]): Seq[Double] = {
     val norm = math.sqrt(vector.map(x => x * x).sum)
     vector.map(_ / norm)
   }