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
  timeout: Duration = Duration.fromSeconds(60)
) extends zio.langchain.core.config.EmbeddingConfig

/**
 * Companion object for OpenAIEmbeddingConfig.
 */
object OpenAIEmbeddingConfig:
  /**
   * Creates an OpenAIEmbeddingConfig from environment variables.
   */
  def fromEnv: OpenAIEmbeddingConfig = 
    OpenAIEmbeddingConfig(
      apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
      model = sys.env.getOrElse("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
      organizationId = sys.env.get("OPENAI_ORG_ID"),
      timeout = Duration.fromMillis(
        sys.env.getOrElse("OPENAI_TIMEOUT_MS", "60000").toLong
      )
    )
  
  /**
   * Creates a ZLayer that provides an OpenAIEmbeddingConfig from environment variables.
   */
  val layer: ULayer[OpenAIEmbeddingConfig] = ZLayer.succeed(fromEnv)

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
  
    
    // Create and send request
    ZIO.scoped {
      for {
        // Get client
        client <- ZIO.service[Client].provide(Client.default)
        
        // Parse URL
        url <- ZIO.fromEither(URL.decode(apiUrl))
                 .orElseFail(new RuntimeException(s"Invalid URL: $apiUrl"))
        body = Body.fromString(jsonBody)
        baseHeaders = Headers(
          Header.ContentType(MediaType.application.json),
          Header.Authorization.Bearer(config.apiKey)
        )
        headers = config.organizationId match {
          case Some(orgId) => baseHeaders ++ Headers(Header.Custom("OpenAI-Organization", orgId))
          case None => baseHeaders
        }
        // Create request
        request = Request.post(
          body = body,
          url = url
        )
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
    yield Embedding(embData.embedding.toVector)
    
    result.mapError(err => EmbeddingError(err))
  
  /**
   * Generates embeddings for multiple texts.
   *
   * @param texts The sequence of texts to embed
   * @return A ZIO effect that produces a sequence of Embeddings or fails with an EmbeddingError
   */
  override def embedAll(texts: Seq[String]): ZIO[Any, EmbeddingError, Seq[Embedding]] =
    if (texts.isEmpty) ZIO.succeed(Seq.empty)
    else
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
        
        // Extract embeddings and sort by index
        sortedEmbeddings = respObj.data
          .sortBy(_.index)
          .map(data => Embedding(data.embedding.toVector))
      yield sortedEmbeddings
      
      result.mapError(err => EmbeddingError(err))

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
  def make(config: OpenAIEmbeddingConfig): UIO[OpenAIEmbedding] =
    ZIO.succeed(new OpenAIEmbedding(config))
   
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
