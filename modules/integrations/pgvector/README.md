# ZIO LangChain PostgreSQL/pgvector Integration

This module provides integration with PostgreSQL and the pgvector extension for vector similarity search in ZIO LangChain.

## Features

- Store and retrieve document embeddings in PostgreSQL using pgvector
- Support for different distance metrics (cosine, L2, inner product)
- Connection pooling for efficient database access
- Proper resource management with ZIO scoped layers
- Comprehensive error handling

## Prerequisites

- PostgreSQL 12+ with pgvector extension installed
- Java 11+
- Scala 3.3+
- ZIO 2.x

## Installation

To install the pgvector extension in PostgreSQL:

```sql
CREATE EXTENSION vector;
```

## Usage

### Configuration

Configure the PostgreSQL connection and pgvector settings:

```scala
import zio.*
import zio.langchain.integrations.pgvector.PgVectorConfig

val config = PgVectorConfig(
  host = "localhost",
  port = 5432,
  database = "langchain",
  username = "postgres",
  password = "password",
  schema = "public",
  table = "document_embeddings",
  dimension = 1536,  // OpenAI embedding dimension
  distanceType = "cosine"  // Options: cosine, l2, inner
)
```

Or load from environment variables:

```scala
val configLayer = ZLayer.fromZIO(
  PgVectorConfig.fromEnv.mapError(err => new RuntimeException(err))
)
```

### Storing Documents

```scala
import zio.*
import zio.langchain.core.domain.Document
import zio.langchain.integrations.openai.OpenAIEmbedding
import zio.langchain.integrations.pgvector.PgVectorStore

val program = for {
  // Get the PgVectorStore service
  store <- ZIO.service[PgVectorStore]
  
  // Create some documents
  documents = Seq(
    Document(
      id = "doc1",
      content = "This is a sample document about ZIO.",
      metadata = Map("source" -> "example", "category" -> "programming")
    ),
    Document(
      id = "doc2",
      content = "PostgreSQL is a powerful database system.",
      metadata = Map("source" -> "example", "category" -> "database")
    )
  )
  
  // Add documents to the store
  _ <- store.addDocuments(documents)
    .tapError(err => Console.printLine(s"Error: ${err.getMessage}"))
    .retry(Schedule.exponential(1.second) && Schedule.recurs(3))
} yield ()

// Provide the necessary dependencies
program.provide(
  OpenAIEmbedding.live,
  PgVectorConfig.layer,
  PgVectorStore.scoped
)
```

### Retrieving Documents

```scala
import zio.*
import zio.langchain.integrations.pgvector.PgVectorStore

val program = for {
  // Get the PgVectorStore service
  store <- ZIO.service[PgVectorStore]
  
  // Query for similar documents
  query = "Tell me about ZIO"
  results <- store.retrieveWithScores(query, maxResults = 5)
  
  // Display results
  _ <- Console.printLine(s"Found ${results.length} results:")
  _ <- ZIO.foreach(results.zipWithIndex) { case ((doc, score), i) =>
    Console.printLine(s"Result ${i + 1} (score: ${"%.4f".format(score)}): ${doc.content}")
  }
} yield ()

// Provide the necessary dependencies
program.provide(
  OpenAIEmbedding.live,
  PgVectorConfig.layer,
  PgVectorStore.scoped
)
```

## Error Handling

The pgvector integration provides specific error types for better error handling:

```scala
import zio.langchain.core.errors.PgVectorError

// Handle specific error types
store.retrieveWithScores(query)
  .catchSome {
    case e if e.getMessage.contains("Connection error") =>
      // Handle connection errors
      ZIO.fail(new RuntimeException("Database connection failed"))
    case e if e.getMessage.contains("Dimension mismatch") =>
      // Handle dimension mismatch errors
      ZIO.fail(new RuntimeException("Embedding dimension mismatch"))
  }
```

## Environment Variables

The following environment variables can be used to configure the pgvector integration:

- `PGVECTOR_HOST`: PostgreSQL host (default: "localhost")
- `PGVECTOR_PORT`: PostgreSQL port (default: 5432)
- `PGVECTOR_DATABASE`: PostgreSQL database name
- `PGVECTOR_USERNAME`: PostgreSQL username
- `PGVECTOR_PASSWORD`: PostgreSQL password
- `PGVECTOR_SCHEMA`: PostgreSQL schema (default: "public")
- `PGVECTOR_TABLE`: PostgreSQL table name
- `PGVECTOR_ID_COLUMN`: Column name for document IDs (default: "id")
- `PGVECTOR_CONTENT_COLUMN`: Column name for document content (default: "content")
- `PGVECTOR_VECTOR_COLUMN`: Column name for vector embeddings (default: "embedding")
- `PGVECTOR_METADATA_COLUMN`: Column name for document metadata (default: "metadata")
- `PGVECTOR_DIMENSION`: Embedding dimension (default: 1536)
- `PGVECTOR_DISTANCE_TYPE`: Distance metric (default: "cosine")
- `PGVECTOR_CONNECTION_POOL_SIZE`: Connection pool size (default: 5)
- `PGVECTOR_TIMEOUT_MS`: Timeout in milliseconds (default: 30000)