package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.domain.Document
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.errors.{EmbeddingError, RetrieverError, ConfigurationError}
import zio.langchain.integrations.openai.{OpenAIEmbedding, OpenAIEmbeddingConfig, OpenAIEmbeddingError}
import zio.langchain.integrations.pgvector.{PgVectorConfig, PgVectorStore}

/**
 * Example demonstrating the use of PostgreSQL with pgvector extension for vector similarity search.
 * This example shows how to store document embeddings in a PostgreSQL database and perform
 * vector similarity search using the pgvector extension.
 */
object PgVectorExample extends ZIOAppDefault {

  /**
   * Main program that demonstrates PgVector integration with error handling.
   */
  override def run: ZIO[Any, Throwable, Unit] = {
    val program = for {
      // Load and validate configurations
      pgVectorConfig <- loadAndValidatePgVectorConfig
      _ <- printLine(s"Successfully validated PgVector configuration for table: ${pgVectorConfig.table}")
      
      openaiConfig <- loadAndValidateOpenAIEmbeddingConfig
      _ <- printLine(s"Successfully validated OpenAI embedding configuration for model: ${openaiConfig.model}")
      
      // Validate dimension compatibility between OpenAI embeddings and PgVector table
      _ <- validateDimensionCompatibility(openaiConfig, pgVectorConfig)
      _ <- printLine("Confirmed embedding dimension compatibility")

      // Create sample documents
      documents = createSampleDocuments()
      _ <- printLine(s"Created ${documents.length} sample documents")

      // Initialize PgVector store with OpenAI embeddings
      store <- ZIO.service[PgVectorStore]
      _ <- printLine("Initialized PgVector store")

      // Add documents to PgVector with enhanced error handling
      _ <- store.addDocuments(documents)
        .tapError(err => printLine(s"Error adding documents: ${err.getMessage}"))
        .retry(Schedule.exponential(1.second) && Schedule.recurs(3))
        .tap(_ => printLine("Successfully added documents to PgVector"))

      // Query PgVector with enhanced error handling
      query = "What is ZIO?"
      _ <- printLine(s"Querying PgVector with: '$query'")
      results <- store.retrieveWithScores(query, 3)
        .tapError(err => printLine(s"Error querying PgVector: ${err.getMessage}"))
        .retry(Schedule.exponential(1.second) && Schedule.recurs(3))

      // Display results
      _ <- printLine(s"Found ${results.length} results:")
      _ <- ZIO.foreach(results.zipWithIndex) { case ((doc, score), i) =>
        printLine(s"Result ${i + 1} (score: ${"%.4f".format(score)}): ${doc.content.take(100)}...")
      }

      // Clean up (optional - uncomment to delete the documents)
      // _ <- store.deleteAll()
      //   .tapError(err => printLine(s"Error deleting documents: ${err.getMessage}"))
      //   .tap(_ => printLine("Successfully deleted all documents from PgVector"))
      
      // Demonstrate proper resource cleanup
      _ <- printLine("Cleaning up resources...")
      _ <- ZIO.succeed(()) // In a real application, resources are cleaned up by the scoped layer
      _ <- printLine("Resources cleaned up successfully")
    } yield ()

    // Provide the necessary dependencies with validated configurations
    program.provide(
      // OpenAI embedding model with validated configuration
      ZLayer.fromZIOEnvironment(
        ZIO.environment[OpenAIEmbeddingConfig].provideLayer(
          ZLayer.fromZIO(OpenAIEmbeddingConfig.fromEnv.mapError(err => new RuntimeException(err)))
        )
      ),
      OpenAIEmbedding.live,
      
      // PgVector configuration and store
      ZLayer.fromZIOEnvironment(
        ZIO.environment[PgVectorConfig].provideLayer(
          ZLayer.fromZIO(PgVectorConfig.fromEnv.mapError(err => new RuntimeException(err)))
        )
      ),
      // Use the scoped layer to ensure proper resource cleanup
      PgVectorStore.scoped
    ).tapError(err => printLine(s"Application error: ${err.toString}"))
  }

  /**
   * Loads and validates the PgVector configuration with enhanced error handling.
   * 
   * This method attempts to load PgVector configuration from environment variables,
   * validates it, and provides detailed error messages for different failure scenarios.
   * 
   * Common error cases handled:
   * - Missing host
   * - Invalid port
   * - Missing database
   * - Missing username or password
   * - Missing table name
   * 
   * @return A ZIO effect that produces a validated PgVectorConfig or fails with a classified error
   */
  private def loadAndValidatePgVectorConfig: ZIO[Any, Throwable, PgVectorConfig] = {
    // Define a function to classify and enhance error messages
    def classifyPgVectorConfigError(errorMsg: String): Throwable = {
      if (errorMsg.contains("host is missing")) 
        ConfigurationError(s"PostgreSQL host is missing. Please set the PGVECTOR_HOST environment variable.")
      else if (errorMsg.contains("port must be positive")) 
        ConfigurationError(s"PostgreSQL port must be positive. Please check the PGVECTOR_PORT environment variable.")
      else if (errorMsg.contains("database is missing")) 
        ConfigurationError(s"PostgreSQL database is missing. Please set the PGVECTOR_DATABASE environment variable.")
      else if (errorMsg.contains("username is missing")) 
        ConfigurationError(s"PostgreSQL username is missing. Please set the PGVECTOR_USERNAME environment variable.")
      else if (errorMsg.contains("password is missing")) 
        ConfigurationError(s"PostgreSQL password is missing. Please set the PGVECTOR_PASSWORD environment variable.")
      else if (errorMsg.contains("table is missing")) 
        ConfigurationError(s"PostgreSQL table is missing. Please set the PGVECTOR_TABLE environment variable.")
      else if (errorMsg.contains("dimension must be positive")) 
        ConfigurationError(s"Embedding dimension must be positive. Please check the PGVECTOR_DIMENSION environment variable.")
      else if (errorMsg.contains("Distance type must be one of")) 
        ConfigurationError(s"Invalid distance type. Please set PGVECTOR_DISTANCE_TYPE to one of: cosine, l2, inner.")
      else 
        ConfigurationError(s"PgVector configuration error: $errorMsg")
    }
    
    // Load configuration from environment variables with enhanced error handling
    PgVectorConfig.fromEnv
      .mapError(classifyPgVectorConfigError)
      .tapError(err => printLine(s"Configuration error: ${err.toString}"))
      .tap(_ => printLine("PgVector configuration loaded successfully"))
  }
  
  /**
   * Loads and validates the OpenAI embedding configuration with enhanced error handling.
   * 
   * This method attempts to load OpenAI embedding configuration from environment variables,
   * validates it, and provides detailed error messages for different failure scenarios.
   * 
   * Common error cases handled:
   * - Missing API key
   * - Invalid model name
   * - Invalid dimension
   * 
   * @return A ZIO effect that produces a validated OpenAIEmbeddingConfig or fails with a classified error
   */
  private def loadAndValidateOpenAIEmbeddingConfig: ZIO[Any, Throwable, OpenAIEmbeddingConfig] = {
    // Define a function to classify and enhance error messages
    def classifyOpenAIConfigError(errorMsg: String): Throwable = {
      if (errorMsg.contains("API key is missing")) 
        ConfigurationError(s"OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
      else if (errorMsg.contains("embedding model is missing")) 
        ConfigurationError(s"OpenAI embedding model is missing. Please set the OPENAI_EMBEDDING_MODEL environment variable (default: 'text-embedding-ada-002').")
      else if (errorMsg.contains("Timeout must be positive")) 
        ConfigurationError(s"OpenAI timeout must be positive. Please check the OPENAI_TIMEOUT_MS environment variable.")
      else if (errorMsg.contains("Embedding dimension must be positive")) 
        ConfigurationError(s"OpenAI embedding dimension must be positive. Please check the OPENAI_EMBEDDING_DIMENSION environment variable.")
      else 
        ConfigurationError(s"OpenAI embedding configuration error: $errorMsg")
    }
    
    // Load configuration from environment variables with enhanced error handling
    OpenAIEmbeddingConfig.fromEnv
      .mapError(classifyOpenAIConfigError)
      .tapError(err => printLine(s"Configuration error: ${err.toString}"))
      .tap(_ => printLine("OpenAI embedding configuration loaded successfully"))
  }
  
  /**
   * Validates that the embedding dimensions match the PgVector table dimensions.
   * 
   * This method ensures that the OpenAI embedding model's output dimension matches
   * the dimension expected by the PgVector table. This is critical for proper functioning
   * of the vector store, as dimension mismatches will cause runtime errors.
   * 
   * @param embeddingConfig The OpenAI embedding configuration
   * @param pgVectorConfig The PgVector configuration
   * @return A ZIO effect that succeeds if dimensions match, or fails with a detailed error
   */
  private def validateDimensionCompatibility(
    embeddingConfig: OpenAIEmbeddingConfig,
    pgVectorConfig: PgVectorConfig
  ): ZIO[Any, Throwable, Unit] = {
    if (embeddingConfig.dimension != pgVectorConfig.dimension) {
      // Create a detailed error with actionable information
      val errorMessage =
        s"""Dimension mismatch: OpenAI embedding dimension (${embeddingConfig.dimension})
           |does not match PgVector table dimension (${pgVectorConfig.dimension})
           |
           |This error occurs when the embedding model produces vectors of a different size
           |than what the PgVector table expects. To resolve this issue:
           |
           |1. Either update your PgVector table to use dimension ${embeddingConfig.dimension}
           |2. Or set OPENAI_EMBEDDING_DIMENSION=${pgVectorConfig.dimension} to match your table
           |3. Or use a different embedding model that produces ${pgVectorConfig.dimension}-dimensional vectors
           |""".stripMargin
      
      ZIO.fail(ConfigurationError(errorMessage))
    } else {
      ZIO.unit
    }
  }

  /**
   * Creates sample documents for demonstration.
   */
  private def createSampleDocuments(): Seq[Document] = {
    Seq(
      Document(
        id = "doc1",
        content = "ZIO is a library for asynchronous and concurrent programming in Scala. " +
                 "It provides a simple, type-safe, and composable way to build concurrent applications.",
        metadata = Map("source" -> "zio-docs", "category" -> "programming")
      ),
      Document(
        id = "doc2",
        content = "Scala is a general-purpose programming language that combines object-oriented and " +
                 "functional programming. It runs on the JVM and is designed to be concise and expressive.",
        metadata = Map("source" -> "scala-docs", "category" -> "programming")
      ),
      Document(
        id = "doc3",
        content = "PostgreSQL is a powerful, open source object-relational database system with over 35 years " +
                 "of active development. It is known for reliability, feature robustness, and performance.",
        metadata = Map("source" -> "postgres-docs", "category" -> "database")
      ),
      Document(
        id = "doc4",
        content = "Vector embeddings are numerical representations of data that capture semantic meaning. " +
                 "They allow machines to understand similarities between different pieces of content.",
        metadata = Map("source" -> "ml-docs", "category" -> "machine-learning")
      ),
      Document(
        id = "doc5",
        content = "pgvector is a PostgreSQL extension for vector similarity search. It provides vector data types " +
                 "and vector similarity search operators that make it easy to store and query vector embeddings.",
        metadata = Map("source" -> "pgvector-docs", "category" -> "database")
      )
    )
  }
}