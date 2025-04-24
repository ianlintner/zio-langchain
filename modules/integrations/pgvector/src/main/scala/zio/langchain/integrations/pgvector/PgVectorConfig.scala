package zio.langchain.integrations.pgvector

import zio.*
import zio.langchain.core.config.EmbeddingConfig

/**
 * Configuration for PostgreSQL with pgvector extension.
 *
 * @param host The PostgreSQL server host
 * @param port The PostgreSQL server port
 * @param database The database name
 * @param username The username for authentication
 * @param password The password for authentication
 * @param schema The schema name (default: "public")
 * @param table The table name for storing vectors
 * @param idColumn The column name for document IDs (default: "id")
 * @param contentColumn The column name for document content (default: "content")
 * @param vectorColumn The column name for vector embeddings (default: "embedding")
 * @param metadataColumn The column name for document metadata (default: "metadata")
 * @param dimension The dimension of the embeddings (default: 1536 for OpenAI embeddings)
 * @param distanceType The distance metric to use for similarity search (default: "cosine")
 * @param connectionPoolSize The size of the connection pool (default: 5)
 * @param timeout The timeout for database operations
 * @param model The embedding model name
 */
case class PgVectorConfig(
  host: String,
  port: Int,
  database: String,
  username: String,
  password: String,
  schema: String = "public",
  table: String,
  idColumn: String = "id",
  contentColumn: String = "content",
  vectorColumn: String = "embedding",
  metadataColumn: String = "metadata",
  dimension: Int = 1536,
  distanceType: String = "cosine",
  connectionPoolSize: Int = 5,
  timeout: Duration = Duration.fromSeconds(30),
  override val model: String = "text-embedding-ada-002"
) extends EmbeddingConfig {
  /**
   * Validates the configuration.
   *
   * @return Either an error message or the validated configuration
   */
  def validate: Either[String, PgVectorConfig] = {
    if (host.trim.isEmpty) Left("PostgreSQL host is missing or empty")
    else if (port <= 0) Left("PostgreSQL port must be positive")
    else if (database.trim.isEmpty) Left("PostgreSQL database is missing or empty")
    else if (username.trim.isEmpty) Left("PostgreSQL username is missing or empty")
    else if (password.trim.isEmpty) Left("PostgreSQL password is missing or empty")
    else if (table.trim.isEmpty) Left("PostgreSQL table is missing or empty")
    else if (dimension <= 0) Left("Embedding dimension must be positive")
    else if (!Seq("cosine", "l2", "inner").contains(distanceType.toLowerCase))
      Left("Distance type must be one of: cosine, l2, inner")
    else if (connectionPoolSize <= 0) Left("Connection pool size must be positive")
    else if (timeout.toMillis <= 0) Left("Timeout must be positive")
    else Right(this)
  }

  /**
   * Returns the JDBC URL for connecting to the PostgreSQL database.
   */
  def jdbcUrl: String = s"jdbc:postgresql://$host:$port/$database"
}

/**
 * Companion object for PgVectorConfig.
 */
object PgVectorConfig {
  /**
   * Creates a PgVectorConfig from environment variables with validation.
   */
  def fromEnv: ZIO[Any, String, PgVectorConfig] =
    ZIO.attempt {
      PgVectorConfig(
        host = sys.env.getOrElse("PGVECTOR_HOST", ""),
        port = sys.env.get("PGVECTOR_PORT").map(_.toInt).getOrElse(5432),
        database = sys.env.getOrElse("PGVECTOR_DATABASE", ""),
        username = sys.env.getOrElse("PGVECTOR_USERNAME", ""),
        password = sys.env.getOrElse("PGVECTOR_PASSWORD", ""),
        schema = sys.env.getOrElse("PGVECTOR_SCHEMA", "public"),
        table = sys.env.getOrElse("PGVECTOR_TABLE", ""),
        idColumn = sys.env.getOrElse("PGVECTOR_ID_COLUMN", "id"),
        contentColumn = sys.env.getOrElse("PGVECTOR_CONTENT_COLUMN", "content"),
        vectorColumn = sys.env.getOrElse("PGVECTOR_VECTOR_COLUMN", "embedding"),
        metadataColumn = sys.env.getOrElse("PGVECTOR_METADATA_COLUMN", "metadata"),
        dimension = sys.env.get("PGVECTOR_DIMENSION").map(_.toInt).getOrElse(1536),
        distanceType = sys.env.getOrElse("PGVECTOR_DISTANCE_TYPE", "cosine"),
        connectionPoolSize = sys.env.get("PGVECTOR_CONNECTION_POOL_SIZE").map(_.toInt).getOrElse(5),
        timeout = Duration.fromMillis(
          sys.env.getOrElse("PGVECTOR_TIMEOUT_MS", "30000").toLong
        )
      )
    }.catchAll(ex => ZIO.fail(s"Error creating PgVectorConfig: ${ex.getMessage}"))
     .flatMap(config => ZIO.fromEither(config.validate))
  
  /**
   * Creates a ZLayer that provides a validated PgVectorConfig from environment variables.
   */
  val layer: ZLayer[Any, String, PgVectorConfig] = ZLayer.fromZIO(fromEnv)
}