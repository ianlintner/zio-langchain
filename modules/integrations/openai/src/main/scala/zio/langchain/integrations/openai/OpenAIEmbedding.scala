package zio.langchain.integrations.openai

import zio.*
import zio.json.*
import zio.http.*

import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * Configuration for OpenAI embedding models.
 *
 * @param apiKey The OpenAI API key
 * @param model The model identifier (e.g., "text-embedding-ada-002")
 * @param organizationId The OpenAI organization ID (optional)
 * @param timeout The timeout for API requests
 */
case class OpenAIEmbeddingConfig(
  apiKey: String,
  model: String = "text-embedding-ada-002",
  organizationId: Option[String] = None,
  timeout: Duration = Duration.fromSeconds(60),
  dimension: Int = 1536 // Default dimension for text-embedding-ada-002
) extends zio.langchain.core.config.EmbeddingConfig {
  /**
   * Validates the configuration.
   *
   * @return Either an error message or the validated configuration
   */
  def validate: Either[String, OpenAIEmbeddingConfig] = {
    if (apiKey.trim.isEmpty) Left("OpenAI API key is missing or empty")
    else if (model.trim.isEmpty) Left("OpenAI embedding model is missing or empty")
    else if (timeout.toMillis <= 0) Left("Timeout must be positive")
    else if (dimension <= 0) Left("Embedding dimension must be positive")
    else Right(this)
  }
}

/**
 * Companion object for OpenAIEmbeddingConfig.
 */
object OpenAIEmbeddingConfig:
  /**
   * Creates an OpenAIEmbeddingConfig from environment variables with validation.
   */
  def fromEnv: ZIO[Any, String, OpenAIEmbeddingConfig] =
    ZIO.attempt {
      OpenAIEmbeddingConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = sys.env.getOrElse("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        organizationId = sys.env.get("OPENAI_ORG_ID"),
        timeout = Duration.fromMillis(
          sys.env.getOrElse("OPENAI_TIMEOUT_MS", "60000").toLong
        ),
        dimension = sys.env.get("OPENAI_EMBEDDING_DIMENSION").map(_.toInt).getOrElse(1536)
      )
    }.catchAll(ex => ZIO.fail(s"Error creating OpenAIEmbeddingConfig: ${ex.getMessage}"))
     .flatMap(config => ZIO.fromEither(config.validate))
  
  /**
   * Creates a ZLayer that provides a validated OpenAIEmbeddingConfig from environment variables.
   */
  val layer: ZLayer[Any, String, OpenAIEmbeddingConfig] = ZLayer.fromZIO(fromEnv)

/**
 * Internal API models for OpenAI embedding API
 */
private object OpenAIApi:
  // Case classes for API requests and responses
  case class EmbeddingRequestSingle(
    model: String,
    input: String
  )
  
  object EmbeddingRequestSingle:
    given JsonEncoder[EmbeddingRequestSingle] = DeriveJsonEncoder.gen[EmbeddingRequestSingle]
    
  case class EmbeddingRequestBatch(
    model: String,
    input: List[String]
  )
  
  object EmbeddingRequestBatch:
    given JsonEncoder[EmbeddingRequestBatch] = DeriveJsonEncoder.gen[EmbeddingRequestBatch]
  
  case class EmbeddingData(
    index: Int,
    embedding: List[Float],
    `object`: String
  )
  
  object EmbeddingData:
    given JsonDecoder[EmbeddingData] = DeriveJsonDecoder.gen[EmbeddingData]
  
  case class Usage(
    prompt_tokens: Int,
    total_tokens: Int
  )
  
  object Usage:
    given JsonDecoder[Usage] = DeriveJsonDecoder.gen[Usage]
  
  case class EmbeddingResponse(
    data: List[EmbeddingData],
    model: String,
    `object`: String,
    usage: Usage
  )
  
  object EmbeddingResponse:
    given JsonDecoder[EmbeddingResponse] = DeriveJsonDecoder.gen[EmbeddingResponse]

/**
 * Implementation of the EmbeddingModel interface for OpenAI models.
 * This version uses ZIO HTTP to connect to OpenAI API.
 */
class OpenAIEmbedding(config: OpenAIEmbeddingConfig) extends EmbeddingModel:
  import OpenAIApi.*
  
  private val apiUrl = "https://api.openai.com/v1/embeddings"

  /**
   * Makes an HTTP request to the OpenAI API using ZIO HTTP
   */
  private def makeRequest(jsonBody: String): ZIO[Any, Throwable, String] = {
    // Create headers
    val headers = Headers(
      Header.ContentType(MediaType.application.json),
      Header.Authorization.Bearer(config.apiKey)
    ) ++ (config.organizationId match {
      case Some(orgId) => Headers(Header.Custom("OpenAI-Organization", orgId))
      case None => Headers.empty
    })
    
    // Create request body
    val body = Body.fromString(jsonBody)
    
    // Create and send request
    ZIO.scoped {
      for {
        // Get client
        client <- ZIO.service[Client].provide(Client.default)
        
        // Parse URL
        url <- ZIO.fromEither(URL.decode(apiUrl))
                 .orElseFail(new RuntimeException(s"Invalid URL: $apiUrl"))
        
        // Create request
        request = Request.post(body, url)
        
        // Send request
        response <- client.request(request)
                      .timeoutFail(new RuntimeException("Request timed out"))(config.timeout)
        
        // Check for errors
        _ <- ZIO.when(response.status.isError) {
          response.body.asString.flatMap { body =>
            ZIO.fail(new RuntimeException(s"OpenAI API error: ${response.status.code}, body: $body"))
          }
        }
        
        // Get response body
        responseBody <- response.body.asString
      } yield responseBody
    }
  }

  /**
   * Generates an embedding for a single text.
   *
   * @param text The text to embed
   * @return A ZIO effect that produces an Embedding or fails with an EmbeddingError
   */
  override def embed(text: String): ZIO[Any, EmbeddingError, Embedding] =
    if (text.trim.isEmpty) {
      ZIO.fail(OpenAIEmbeddingError.invalidRequestError("Empty text provided for embedding"))
    } else {
      val requestBody = EmbeddingRequestSingle(
        model = config.model,
        input = text
      )
      
      val jsonBody = requestBody.toJson
      
      val result = for
        // Call OpenAI API
        respBody <- makeRequest(jsonBody)
        
        // Parse the response
        respObj <- ZIO.fromEither(respBody.fromJson[EmbeddingResponse])
          .mapError(err => new RuntimeException(s"Failed to parse response: $err"))
        
        // Extract the embedding
        embData <- ZIO.attempt(respObj.data.head)
          .catchAll(err => ZIO.fail(new RuntimeException(s"Failed to extract embedding data: ${err.getMessage}")))
        
        // Validate embedding dimension if needed
        _ <- ZIO.when(config.dimension > 0 && embData.embedding.size != config.dimension) {
          ZIO.fail(new RuntimeException(
            s"Embedding dimension mismatch: expected ${config.dimension}, got ${embData.embedding.size}"
          ))
        }
      yield Embedding(embData.embedding.toVector)
      
      result.mapError {
        case e: java.net.ConnectException =>
          OpenAIEmbeddingError.serverError(s"Connection error: ${e.getMessage}")
        case e: java.net.SocketTimeoutException =>
          OpenAIEmbeddingError.timeoutError(s"Socket timeout: ${e.getMessage}")
        case e: java.io.IOException =>
          OpenAIEmbeddingError.serverError(s"IO error: ${e.getMessage}")
        case e: RuntimeException if e.getMessage.contains("dimension mismatch") =>
          OpenAIEmbeddingError.dimensionMismatchError(config.dimension, e.getMessage.split("got ")(1).toInt)
        case e =>
          EmbeddingError(e)
      }
    }
  
  /**
   * Generates embeddings for multiple texts.
   *
   * @param texts The sequence of texts to embed
   * @return A ZIO effect that produces a sequence of Embeddings or fails with an EmbeddingError
   */
  override def embedAll(texts: Seq[String]): ZIO[Any, EmbeddingError, Seq[Embedding]] =
    if (texts.isEmpty) ZIO.succeed(Seq.empty)
    else if (texts.exists(_.trim.isEmpty)) {
      ZIO.fail(OpenAIEmbeddingError.invalidRequestError("One or more empty texts provided for embedding"))
    } else {
      val requestBody = EmbeddingRequestBatch(
        model = config.model,
        input = texts.toList
      )
      
      val jsonBody = requestBody.toJson
      
      val result = for
        // Call OpenAI API
        respBody <- makeRequest(jsonBody)
        
        // Parse the response
        respObj <- ZIO.fromEither(respBody.fromJson[EmbeddingResponse])
          .mapError(err => new RuntimeException(s"Failed to parse response: $err"))
        
        // Validate response data length matches input length
        _ <- ZIO.when(respObj.data.length != texts.length) {
          ZIO.fail(new RuntimeException(
            s"Expected ${texts.length} embeddings, but got ${respObj.data.length}"
          ))
        }
        
        // Validate embedding dimensions if needed
        _ <- ZIO.foreach(respObj.data) { embData =>
          ZIO.when(config.dimension > 0 && embData.embedding.size != config.dimension) {
            ZIO.fail(new RuntimeException(
              s"Embedding dimension mismatch: expected ${config.dimension}, got ${embData.embedding.size}"
            ))
          }
        }
        
        // Extract embeddings and sort by index
        sortedEmbeddings = respObj.data
          .sortBy(_.index)
          .map(data => Embedding(data.embedding.toVector))
      yield sortedEmbeddings
      
      result.mapError {
        case e: java.net.ConnectException =>
          OpenAIEmbeddingError.serverError(s"Connection error: ${e.getMessage}")
        case e: java.net.SocketTimeoutException =>
          OpenAIEmbeddingError.timeoutError(s"Socket timeout: ${e.getMessage}")
        case e: java.io.IOException =>
          OpenAIEmbeddingError.serverError(s"IO error: ${e.getMessage}")
        case e: RuntimeException if e.getMessage.contains("dimension mismatch") =>
          OpenAIEmbeddingError.dimensionMismatchError(config.dimension, e.getMessage.split("got ")(1).toInt)
        case e =>
          EmbeddingError(e)
      }
    }

/**
 * Companion object for OpenAIEmbedding.
 */
object OpenAIEmbedding:
  /**
   * Creates an OpenAIEmbedding from an OpenAIEmbeddingConfig.
   *
   * @param config The OpenAI embedding configuration
   * @return A ZIO effect that produces an OpenAIEmbedding
   */
  def make(config: OpenAIEmbeddingConfig): UIO[OpenAIEmbedding] = {
    ZIO.succeed(new OpenAIEmbedding(config))
  }
  /**
   * Creates a ZLayer that provides an EmbeddingModel implementation using OpenAI.
   *
   * @return A ZLayer that requires an OpenAIEmbeddingConfig and provides an EmbeddingModel
   */
  val live: ZLayer[OpenAIEmbeddingConfig, Nothing, EmbeddingModel] =
    ZLayer {
      for
        config <- ZIO.service[OpenAIEmbeddingConfig]
        embedding <- make(config)
      yield embedding
  }
    
/**
 * OpenAI Embedding API specific error helpers
 */
object OpenAIEmbeddingError {
  // Create wrapper methods to convert OpenAI embedding errors to EmbeddingError
  def authenticationError(message: String): EmbeddingError =
    EmbeddingError(new RuntimeException(s"Authentication error: $message"), "OpenAI embedding authentication failed")
  
  def rateLimitError(message: String): EmbeddingError =
    EmbeddingError(new RuntimeException(s"Rate limit exceeded: $message"), "OpenAI embedding rate limit exceeded")
  
  def serverError(message: String): EmbeddingError =
    EmbeddingError(new RuntimeException(s"OpenAI server error: $message"), "OpenAI embedding server error")
  
  def invalidRequestError(message: String): EmbeddingError =
    EmbeddingError(new RuntimeException(s"Invalid request: $message"), "Invalid request to OpenAI embedding API")
  
  def timeoutError(message: String): EmbeddingError =
    EmbeddingError(new RuntimeException(s"Request timed out: $message"), "OpenAI embedding request timed out")
  
  def dimensionMismatchError(expected: Int, actual: Int): EmbeddingError =
    EmbeddingError(new RuntimeException(s"Dimension mismatch: expected $expected, got $actual"),
                  "Vector dimension mismatch in OpenAI embedding")
  
  def unknownError(cause: Throwable): EmbeddingError =
    EmbeddingError(cause, "Unknown OpenAI embedding error")
}
    
  /**
   * Creates a ZLayer that provides an EmbeddingModel implementation using OpenAI with configuration from environment variables.
   *
   * @return A ZLayer that provides an EmbeddingModel
   */
  val layer: ZLayer[Any, String, EmbeddingModel] =
    OpenAIEmbeddingConfig.layer >>> OpenAIEmbedding.live
