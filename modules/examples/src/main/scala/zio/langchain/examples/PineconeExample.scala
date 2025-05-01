package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.domain.Document
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.errors.{EmbeddingError, RetrieverError, ConfigurationError}
import zio.langchain.integrations.openai.{OpenAIEmbedding, OpenAIEmbeddingConfig, OpenAIEmbeddingError}
import zio.langchain.integrations.pinecone.{PineconeConfig, PineconeStore}

/**
 * Example demonstrating the use of Pinecone vector store with proper error handling
 * and configuration validation.
 */
object PineconeExample extends ZIOAppDefault {

  /**
   * Main program that demonstrates Pinecone integration with error handling.
   */
  override def run: ZIO[Any, Throwable, Unit] = {
    val program = for {
      // Load and validate configurations
      pineconeConfig <- loadAndValidatePineconeConfig
      _ <- printLine(s"Successfully validated Pinecone configuration for index: ${pineconeConfig.indexName}")
      
      openaiConfig <- loadAndValidateOpenAIEmbeddingConfig
      _ <- printLine(s"Successfully validated OpenAI embedding configuration for model: ${openaiConfig.model}")
      
      // Validate dimension compatibility between OpenAI embeddings and Pinecone index
      _ <- validateDimensionCompatibility(openaiConfig, pineconeConfig)
      _ <- printLine("Confirmed embedding dimension compatibility")

      // Create sample documents
      documents = createSampleDocuments()
      _ <- printLine(s"Created ${documents.length} sample documents")

      // Initialize Pinecone store with OpenAI embeddings
      store <- ZIO.service[PineconeStore]
      _ <- printLine("Initialized Pinecone store")

      // Add documents to Pinecone with enhanced error handling
      _ <- store.addDocuments(documents)
        .tapError(err => printLine(s"Error adding documents: ${err.getMessage}"))
        .retry(Schedule.exponential(1.second) && Schedule.recurs(3))
        .tap(_ => printLine("Successfully added documents to Pinecone"))

      // Query Pinecone with enhanced error handling
      query = "What is ZIO?"
      _ <- printLine(s"Querying Pinecone with: '$query'")
      results <- store.retrieveWithScores(query, 3)
        .tapError(err => printLine(s"Error querying Pinecone: ${err.getMessage}"))
        .retry(Schedule.exponential(1.second) && Schedule.recurs(3))

      // Display results
      _ <- printLine(s"Found ${results.length} results:")
      _ <- ZIO.foreach(results.zipWithIndex) { case ((doc, score), i) =>
        printLine(s"Result ${i + 1} (score: ${"%.4f".format(score)}): ${doc.content.take(100)}...")
      }

      // Clean up (optional - uncomment to delete the documents)
      // _ <- store.deleteAll()
      //   .tapError(err => printLine(s"Error deleting documents: ${err.getMessage}"))
      //   .tap(_ => printLine("Successfully deleted all documents from Pinecone"))
      
      // Demonstrate proper resource cleanup
      _ <- printLine("Cleaning up resources...")
      _ <- ZIO.succeed(()) // In a real application, you would close connections or release resources here
      _ <- printLine("Resources cleaned up successfully")
    } yield ()

    // Provide the necessary dependencies with validated configurations
    program.provide(
      // HTTP Client dependency required by all API integrations
      Client.default,
      
      // OpenAI embedding model with validated configuration
      ZLayer.fromZIOEnvironment(
        ZIO.environment[OpenAIEmbeddingConfig].provideLayer(
          ZLayer.fromZIO(OpenAIEmbeddingConfig.fromEnv.mapError(err => new RuntimeException(err)))
        )
      ),
      OpenAIEmbedding.live,
      
      // Pinecone configuration and store
      ZLayer.fromZIOEnvironment(
        ZIO.environment[PineconeConfig].provideLayer(
          ZLayer.fromZIO(PineconeConfig.fromEnv.mapError(err => new RuntimeException(err)))
        )
      ),
      PineconeStore.liveStore
    ).tapError(err => printLine(s"Application error: ${err.toString}"))
  }

  /**
   * Loads and validates the Pinecone configuration with enhanced error handling.
   * 
   * This method attempts to load Pinecone configuration from environment variables,
   * validates it, and provides detailed error messages for different failure scenarios.
   * 
   * Common error cases handled:
   * - Missing API key
   * - Missing environment
   * - Missing project ID
   * - Missing index name
   * 
   * @return A ZIO effect that produces a validated PineconeConfig or fails with a classified error
   */
  private def loadAndValidatePineconeConfig: ZIO[Any, Throwable, PineconeConfig] = {
    // Define a function to classify and enhance error messages
    def classifyPineconeConfigError(errorMsg: String): Throwable = {
      if (errorMsg.contains("API key is missing")) 
        ConfigurationError(s"Pinecone API key is missing. Please set the PINECONE_API_KEY environment variable.")
      else if (errorMsg.contains("environment is missing")) 
        ConfigurationError(s"Pinecone environment is missing. Please set the PINECONE_ENVIRONMENT environment variable (e.g., 'us-west1-gcp').")
      else if (errorMsg.contains("project ID is missing")) 
        ConfigurationError(s"Pinecone project ID is missing. Please set the PINECONE_PROJECT_ID environment variable.")
      else if (errorMsg.contains("index name is missing")) 
        ConfigurationError(s"Pinecone index name is missing. Please set the PINECONE_INDEX_NAME environment variable.")
      else 
        ConfigurationError(s"Pinecone configuration error: $errorMsg")
    }
    
    // Load configuration from environment variables with enhanced error handling
    PineconeConfig.fromEnv
      .mapError(classifyPineconeConfigError)
      .tapError(err => printLine(s"Configuration error: ${err.toString}"))
      .tap(_ => printLine("Pinecone configuration loaded successfully"))
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
   * Validates that the embedding dimensions match the Pinecone index dimensions.
   * 
   * This method ensures that the OpenAI embedding model's output dimension matches
   * the dimension expected by the Pinecone index. This is critical for proper functioning
   * of the vector store, as dimension mismatches will cause runtime errors.
   * 
   * @param embeddingConfig The OpenAI embedding configuration
   * @param pineconeConfig The Pinecone configuration
   * @return A ZIO effect that succeeds if dimensions match, or fails with a detailed error
   */
  private def validateDimensionCompatibility(
    embeddingConfig: OpenAIEmbeddingConfig,
    pineconeConfig: PineconeConfig
  ): ZIO[Any, Throwable, Unit] = {
    if (embeddingConfig.dimension != pineconeConfig.dimension) {
      // Create a detailed error with actionable information
      val errorMessage =
        s"""Dimension mismatch: OpenAI embedding dimension (${embeddingConfig.dimension})
           |does not match Pinecone index dimension (${pineconeConfig.dimension})
           |
           |This error occurs when the embedding model produces vectors of a different size
           |than what the Pinecone index expects. To resolve this issue:
           |
           |1. Either update your Pinecone index to use dimension ${embeddingConfig.dimension}
           |2. Or set OPENAI_EMBEDDING_DIMENSION=${pineconeConfig.dimension} to match your index
           |3. Or use a different embedding model that produces ${pineconeConfig.dimension}-dimensional vectors
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
        content = "Pinecone is a vector database that makes it easy to build high-performance vector search " +
                 "applications. It's designed for machine learning and similarity search use cases.",
        metadata = Map("source" -> "pinecone-docs", "category" -> "database")
      ),
      Document(
        id = "doc4",
        content = "Vector embeddings are numerical representations of data that capture semantic meaning. " +
                 "They allow machines to understand similarities between different pieces of content.",
        metadata = Map("source" -> "ml-docs", "category" -> "machine-learning")
      ),
      Document(
        id = "doc5",
        content = "ZIO provides a comprehensive set of tools for building concurrent applications, " +
                 "including fibers (lightweight threads), queues, semaphores, and much more.",
        metadata = Map("source" -> "zio-docs", "category" -> "concurrency")
      )
    )
  }
}
