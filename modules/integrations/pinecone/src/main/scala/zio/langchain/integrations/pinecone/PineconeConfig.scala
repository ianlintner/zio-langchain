package zio.langchain.integrations.pinecone

import zio.*
import zio.langchain.core.config.EmbeddingConfig

/**
 * Configuration for Pinecone vector store.
 *
 * @param apiKey The Pinecone API key
 * @param environment The Pinecone environment (e.g., "us-west1-gcp")
 * @param projectId The Pinecone project ID
 * @param indexName The name of the Pinecone index
 * @param namespace Optional namespace within the index (default: None)
 * @param dimension The dimension of the embeddings (default: 1536 for OpenAI embeddings)
 * @param timeout The timeout for API requests
 */
case class PineconeConfig(
  apiKey: String,
  environment: String,
  projectId: String,
  indexName: String,
  namespace: Option[String] = None,
  dimension: Int = 1536,
  timeout: Duration = Duration.fromSeconds(60),
  override val model: String = "text-embedding-ada-002"
) extends EmbeddingConfig {
  /**
   * Validates the configuration.
   *
   * @return Either an error message or the validated configuration
   */
  def validate: Either[String, PineconeConfig] = {
    if (apiKey.trim.isEmpty) Left("Pinecone API key is missing or empty")
    else if (environment.trim.isEmpty) Left("Pinecone environment is missing or empty")
    else if (projectId.trim.isEmpty) Left("Pinecone project ID is missing or empty")
    else if (indexName.trim.isEmpty) Left("Pinecone index name is missing or empty")
    else Right(this)
  }
}

/**
 * Companion object for PineconeConfig.
 */
object PineconeConfig:
  /**
   * Creates a PineconeConfig from environment variables with validation.
   */
  def fromEnv: ZIO[Any, String, PineconeConfig] =
    ZIO.attempt {
      PineconeConfig(
        apiKey = sys.env.getOrElse("PINECONE_API_KEY", ""),
        environment = sys.env.getOrElse("PINECONE_ENVIRONMENT", ""),
        projectId = sys.env.getOrElse("PINECONE_PROJECT_ID", ""),
        indexName = sys.env.getOrElse("PINECONE_INDEX_NAME", ""),
        namespace = sys.env.get("PINECONE_NAMESPACE"),
        dimension = sys.env.get("PINECONE_DIMENSION").map(_.toInt).getOrElse(1536),
        timeout = Duration.fromMillis(
          sys.env.getOrElse("PINECONE_TIMEOUT_MS", "60000").toLong
        )
      )
    }.catchAll(ex => ZIO.fail(s"Error creating PineconeConfig: ${ex.getMessage}"))
     .flatMap(config => ZIO.fromEither(config.validate))
  
  /**
   * Creates a ZLayer that provides a validated PineconeConfig from environment variables.
   */
  val layer: ZLayer[Any, String, PineconeConfig] = ZLayer.fromZIO(fromEnv)