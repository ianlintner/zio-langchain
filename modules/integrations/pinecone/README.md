# ZIO LangChain Pinecone Integration

This module provides integration with [Pinecone](https://www.pinecone.io/), a vector database for storing and retrieving embeddings for use in retrieval-augmented generation (RAG) applications.

## Features

- Store and retrieve document embeddings in Pinecone
- Similarity search with customizable parameters
- Proper ZIO integration with resource management
- Type-safe configuration
- Efficient connection pooling

## Installation

Add the following dependency to your `build.sbt`:

```scala
libraryDependencies += "dev.zio" %% "zio-langchain-pinecone" % "<version>"
```

## Configuration

You can configure the Pinecone integration using environment variables, a configuration file, or programmatically.

### Environment Variables

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: The Pinecone environment (e.g., "us-west1-gcp")
- `PINECONE_PROJECT_ID`: Your Pinecone project ID
- `PINECONE_INDEX_NAME`: The name of your Pinecone index
- `PINECONE_NAMESPACE`: (Optional) Namespace within the index
- `PINECONE_DIMENSION`: (Optional) Dimension of the embeddings (default: 1536 for OpenAI embeddings)
- `PINECONE_TIMEOUT_MS`: (Optional) Timeout for API requests in milliseconds (default: 60000)

### Configuration File

You can also configure the integration using a configuration file. Add the following to your `application.conf`:

```hocon
pinecone {
  api-key = "your-api-key"
  environment = "your-environment"
  project-id = "your-project-id"
  index-name = "your-index-name"
  namespace = "optional-namespace"
  dimension = 1536
  timeout-ms = 60000
}
```

### Programmatic Configuration

```scala
import zio.langchain.integrations.pinecone.PineconeConfig

val config = PineconeConfig(
  apiKey = "your-api-key",
  environment = "your-environment",
  projectId = "your-project-id",
  indexName = "your-index-name",
  namespace = Some("optional-namespace"),
  dimension = 1536,
  timeout = zio.Duration.fromSeconds(60)
)
```

## Usage

### Basic Usage

```scala
import zio.*
import zio.langchain.core.domain.*
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.integrations.openai.{OpenAIEmbedding, OpenAIEmbeddingConfig}
import zio.langchain.integrations.pinecone.{PineconeStore, PineconeConfig}

object PineconeExample extends ZIOAppDefault {
  override def run = {
    val program = for {
      // Get the services
      pineconeStore <- ZIO.service[PineconeStore]
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Create some documents
      documents = Seq(
        Document(
          id = "doc1",
          content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
          metadata = Map("source" -> "zio-docs")
        ),
        Document(
          id = "doc2",
          content = "Pinecone is a vector database for machine learning applications.",
          metadata = Map("source" -> "pinecone-docs")
        )
      )
      
      // Add documents to Pinecone
      _ <- pineconeStore.addDocuments(documents)
      
      // Query for similar documents
      query = "How does ZIO handle concurrency?"
      results <- pineconeStore.retrieveWithScores(query, maxResults = 5)
      
      // Print results
      _ <- ZIO.foreach(results) { case (doc, score) =>
        Console.printLine(s"Document: ${doc.content} (Score: $score)")
      }
    } yield ()
    
    // Provide the required layers
    program.provide(
      // Pinecone layer
      PineconeStore.liveStore,
      // Pinecone configuration
      ZLayer.succeed(
        PineconeConfig(
          apiKey = sys.env.getOrElse("PINECONE_API_KEY", ""),
          environment = sys.env.getOrElse("PINECONE_ENVIRONMENT", ""),
          projectId = sys.env.getOrElse("PINECONE_PROJECT_ID", ""),
          indexName = sys.env.getOrElse("PINECONE_INDEX_NAME", ""),
          namespace = sys.env.get("PINECONE_NAMESPACE")
        )
      ),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-3-large"
        )
      )
    )
  }
}
```

### Using as a Retriever

The `PineconeStore` implements the `Retriever` interface, so it can be used directly in RAG applications:

```scala
import zio.*
import zio.langchain.core.retriever.Retriever
import zio.langchain.integrations.pinecone.{PineconeStore, PineconeConfig}
import zio.langchain.integrations.openai.{OpenAIEmbedding, OpenAIEmbeddingConfig}

object RAGExample extends ZIOAppDefault {
  override def run = {
    val program = for {
      // Get the retriever
      retriever <- ZIO.service[Retriever]
      
      // Use the retriever in a RAG application
      query = "What is ZIO?"
      documents <- retriever.retrieve(query, maxResults = 3)
      
      // Print retrieved documents
      _ <- ZIO.foreach(documents) { doc =>
        Console.printLine(s"Retrieved: ${doc.content}")
      }
    } yield ()
    
    // Provide the Pinecone retriever
    program.provide(
      // Use PineconeStore as a Retriever
      PineconeStore.live,
      // Pinecone configuration
      ZLayer.succeed(PineconeConfig.fromEnv),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration
      ZLayer.succeed(OpenAIEmbeddingConfig.fromEnv)
    )
  }
}
```

## Advanced Usage

### Custom Similarity Threshold

You can create a custom retriever with a similarity threshold:

```scala
import zio.*
import zio.langchain.core.domain.*
import zio.langchain.core.retriever.Retriever
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.integrations.pinecone.PineconeStore

def createCustomRetriever(
  pineconeStore: PineconeStore,
  embeddingModel: EmbeddingModel,
  similarityThreshold: Float = 0.7f
): Retriever = new Retriever {
  override def retrieve(query: String, maxResults: Int) = {
    for {
      // Get results with scores
      resultsWithScores <- pineconeStore.retrieveWithScores(query, maxResults)
      
      // Filter by similarity threshold
      filteredResults = resultsWithScores
        .filter { case (_, score) => score >= similarityThreshold }
        .map { case (doc, _) => doc }
    } yield filteredResults
  }
}
```

### Metadata Filtering

You can filter results by metadata when using the Pinecone store:

```scala
import zio.*
import zio.langchain.core.domain.*
import zio.langchain.core.retriever.Retriever
import zio.langchain.integrations.pinecone.PineconeStore

def createMetadataFilteredRetriever(
  pineconeStore: PineconeStore,
  metadataKey: String,
  metadataValue: String
): Retriever = new Retriever {
  override def retrieve(query: String, maxResults: Int) = {
    for {
      // Get results with scores
      resultsWithScores <- pineconeStore.retrieveWithScores(query, maxResults * 2)
      
      // Filter by metadata
      filteredResults = resultsWithScores
        .filter { case (doc, _) => 
          doc.metadata.get(metadataKey).contains(metadataValue)
        }
        .take(maxResults)
        .map { case (doc, _) => doc }
    } yield filteredResults
  }
}
```

## Notes

- Make sure your Pinecone index is created with the correct dimension that matches your embedding model.
- For OpenAI embeddings, the dimension is typically 1536.
- The Pinecone integration uses ZIO HTTP for API calls, ensuring proper resource management.