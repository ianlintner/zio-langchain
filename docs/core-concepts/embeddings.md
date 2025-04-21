---
title: Embeddings
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Embeddings

This document explains vector embeddings in ZIO LangChain - how they work, their importance, and how to use them effectively.

## Table of Contents

- [What Are Embeddings?](#what-are-embeddings)
- [Core Embedding Interface](#core-embedding-interface)
- [Working with Embeddings](#working-with-embeddings)
- [Supported Embedding Models](#supported-embedding-models)
- [Vector Similarity](#vector-similarity)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## What Are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. They convert words, sentences, or documents into numerical vectors, allowing:

- Semantic search (finding similar content)
- Retrieval-Augmented Generation (RAG)
- Document clustering and organization
- Conceptual similarity comparisons

Unlike simple keyword matching, embeddings can find content that is conceptually similar even when using different words or phrasing.

## Core Embedding Interface

ZIO LangChain provides a core abstraction for embeddings via the `EmbeddingModel` trait:

```scala
trait EmbeddingModel:
  def embed(text: String): ZIO[Any, EmbeddingError, Embedding]
  def embedAll(texts: Seq[String]): ZIO[Any, EmbeddingError, Seq[Embedding]]
  
  def embedDocument(document: Document): ZIO[Any, EmbeddingError, (Document, Embedding)] =
    embed(document.content).map(embedding => (document, embedding))
    
  def embedDocuments(documents: Seq[Document]): ZIO[Any, EmbeddingError, Seq[(Document, Embedding)]] =
    ZIO.foreachPar(documents)(embedDocument)
```

The `Embedding` type is implemented as an opaque type wrapper around a vector of floating-point values:

```scala
opaque type Embedding = Vector[Float]
object Embedding:
  def apply(values: Vector[Float]): Embedding = values
  extension (e: Embedding)
    def values: Vector[Float] = e
    def dimension: Int = e.size
    def cosineSimilarity(other: Embedding): Float = // implementation
```

## Working with Embeddings

### Creating Embeddings

```scala
import zio.*
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.integrations.openai.OpenAIEmbedding
import zio.langchain.core.domain.Embedding

val program = for
  embeddingModel <- ZIO.service[EmbeddingModel]
  embedding <- embeddingModel.embed("This is a sample text to embed.")
  _ <- Console.printLine(s"Embedding dimension: ${embedding.dimension}")
yield ()

program.provide(
  OpenAIEmbedding.live,
  ZLayer.succeed(OpenAIEmbeddingConfig(...))
)
```

### Processing Multiple Texts

```scala
val texts = Seq(
  "Artificial intelligence",
  "Machine learning",
  "Natural language processing"
)

val program = for
  embeddingModel <- ZIO.service[EmbeddingModel]
  embeddings <- embeddingModel.embedAll(texts)
  _ <- ZIO.foreach(embeddings.zip(texts)) { case (embedding, text) =>
    Console.printLine(s"$text: ${embedding.dimension} dimensions")
  }
yield ()
```

### Embedding Documents

ZIO LangChain provides convenient methods for embedding entire documents:

```scala
import zio.langchain.core.domain.Document

val documents = Seq(
  Document(id = "doc1", content = "ZIO is a functional effect system for Scala."),
  Document(id = "doc2", content = "Embeddings are vector representations of text.")
)

val program = for
  embeddingModel <- ZIO.service[EmbeddingModel]
  embeddedDocs <- embeddingModel.embedDocuments(documents)
yield embeddedDocs
```

## Supported Embedding Models

ZIO LangChain supports embeddings from various providers:

- **OpenAI**: text-embedding-ada-002 and other models
- **HuggingFace**: Various embedding models
- Custom embedding models via the extension API

### OpenAI Embeddings

```scala
import zio.langchain.integrations.openai.OpenAIEmbedding

val embeddingLayer = OpenAIEmbedding.live.provide(
  ZLayer.succeed(
    OpenAIEmbeddingConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = "text-embedding-ada-002"
    )
  )
)
```

## Vector Similarity

The `Embedding` type provides built-in functionality for comparing vectors using cosine similarity:

```scala
val program = for
  embeddingModel <- ZIO.service[EmbeddingModel]
  embedding1 <- embeddingModel.embed("Machine learning models")
  embedding2 <- embeddingModel.embed("Deep learning algorithms")
  similarity = embedding1.cosineSimilarity(embedding2)
  _ <- Console.printLine(s"Similarity score: $similarity")
yield ()
```

Similarity scores range from -1 to 1, where:
- 1 indicates perfect similarity
- 0 indicates orthogonality (unrelated)
- -1 indicates perfect dissimilarity (opposite)

In practice, most embedding similarity scores fall between 0 and 1.

## Usage Examples

### Simple Semantic Search

```scala
def semanticSearch(
  query: String, 
  documents: Seq[Document], 
  topK: Int = 3
): ZIO[EmbeddingModel, EmbeddingError, Seq[Document]] = {
  for
    embeddingModel <- ZIO.service[EmbeddingModel]
    queryEmbedding <- embeddingModel.embed(query)
    docEmbeddings <- embeddingModel.embedDocuments(documents)
    
    // Calculate similarities and sort
    similarities = docEmbeddings.map { case (doc, embedding) =>
      (doc, queryEmbedding.cosineSimilarity(embedding))
    }
    
    // Get top K results
    topResults = similarities
      .sortBy(-_._2)  // Sort by similarity descending
      .take(topK)
      .map(_._1)      // Extract just the documents
  yield topResults
}
```

### Building a RAG System

```scala
def retrieveAndGenerate(
  query: String,
  documents: Seq[Document]
): ZIO[EmbeddingModel with LLM, LangChainError, String] = {
  for
    // Retrieve relevant documents
    relevantDocs <- semanticSearch(query, documents, 3)
      .mapError(e => e: LangChainError)
    
    // Format context
    context = relevantDocs.map(_.content).mkString("\n\n")
    
    // Generate answer with context
    llm <- ZIO.service[LLM]
    prompt = s"""Answer based on the following context:
                |
                |$context
                |
                |Question: $query
                |Answer:""".stripMargin
                
    response <- llm.complete(prompt)
      .mapError(e => e: LangChainError)
  yield response
}
```

## Best Practices

1. **Chunk Documents Appropriately**: Divide large documents into semantically coherent chunks (paragraphs, sections) before embedding.

2. **Normalize Text**: Consistent preprocessing improves embedding quality:
   - Standardize whitespace
   - Handle special characters consistently
   - Consider lowercase for some use cases

3. **Caching**: Embedding computation is expensive - cache results when possible:
   ```scala
   val cachedEmbeddingModel = new EmbeddingModel {
     private val cache = new ConcurrentHashMap[String, Embedding]()
     private val underlying = // real model
     
     override def embed(text: String): ZIO[Any, EmbeddingError, Embedding] =
       ZIO.succeed(Option(cache.get(text)))
         .flatMap {
           case Some(embedding) => ZIO.succeed(embedding)
           case None => underlying.embed(text).tap { embedding =>
             ZIO.succeed(cache.put(text, embedding))
           }
         }
     
     // Implement other methods
   }
   ```

4. **Batch Processing**: Use `embedAll` for multiple texts to take advantage of batching optimizations.

5. **Monitor Dimensions**: Different models produce embeddings with different dimensions. Ensure your system can handle varying dimensions.

6. **Storage Considerations**: Embeddings require significant storage - consider formats and databases specifically designed for vector data.