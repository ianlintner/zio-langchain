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
  timeout: Duration = Duration.fromSeconds(60)
) extends EmbeddingConfig

/**
 * Companion object for PineconeConfig.
 */
object PineconeConfig:
  /**
   * Creates a PineconeConfig from environment variables.
   */
  def fromEnv: PineconeConfig = 
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
  
  /**
   * Creates a ZLayer that provides a PineconeConfig from environment variables.
   */
  val layer: ULayer[PineconeConfig] = ZLayer.succeed(fromEnv)