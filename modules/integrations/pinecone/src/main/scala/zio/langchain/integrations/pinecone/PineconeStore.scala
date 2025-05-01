package zio.langchain.integrations.pinecone

import zio.*
import zio.json.*
import zio.stream.ZStream
import zio.http.{Client, Header, Headers, Method, Body, URL, Request, Response, MediaType}

import zio.langchain.core.retriever.Retriever
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

import java.util.UUID

/**
 * Implementation of a vector store using Pinecone.
 * This class provides methods for storing and retrieving documents using vector embeddings.
 *
 * @param config The Pinecone configuration
 * @param embeddingModel The embedding model to use for generating embeddings
 * @param client The HTTP client to use for API requests
 */
class PineconeStore private (
  config: PineconeConfig,
  embeddingModel: EmbeddingModel,
  client: Client
) extends Retriever:
  import PineconeApi.*
  import zio.langchain.core.errors.PineconeError

  /**
   * The base URL for the Pinecone API.
   */
  private val baseUrl = s"https://${config.indexName}-${config.projectId}.svc.${config.environment}.pinecone.io"

  /**
   * Makes an HTTP request to the Pinecone API.
   */
  private def makeRequest[A](
    path: String,
    method: Method,
    jsonBody: Option[String] = None
  )(using JsonDecoder[A]): ZIO[Any, RetrieverError, A] = {
    val urlString = s"$baseUrl$path"
    
    for {
      // Parse URL
      url <- ZIO.fromEither(URL.decode(urlString))
        .mapError(e => PineconeError.invalidRequestError(s"Invalid URL: $urlString, error: $e"))
      
      // Create headers
      headers = Headers(
        Header.Authorization.Bearer(config.apiKey),
        Header.ContentType(MediaType.application.json)
      )
      
      // Create request based on method
      request = method match {
        case Method.POST =>
          jsonBody match {
            case Some(bodyContent) => 
              Request.post(Body.fromString(bodyContent), url)
                .updateHeaders(_ ++ headers)
            case None => 
              Request.post(Body.empty, url)
                .updateHeaders(_ ++ headers)
          }
        case _ => 
          Request.get(url)
            .updateHeaders(_ ++ headers)
      }
      
      // Execute request
      response <- client.request(request)
        .mapError(e => PineconeError.serverError(s"Connection error: ${e.getMessage}"))
      
      // Process response
      result <- 
        if (response.status.isSuccess) {
          response.body.asString
            .mapError(e => PineconeError.serverError(s"Failed to read response body: ${e.getMessage}"))
            .flatMap { bodyStr =>
              ZIO.fromEither(bodyStr.fromJson[A])
                .mapError(err => PineconeError.invalidRequestError(s"Failed to decode response: $err"))
            }
        } else {
          response.body.asString
            .mapError(e => PineconeError.serverError(s"Failed to read error response: ${e.getMessage}"))
            .flatMap { bodyStr =>
              val error = response.status.code match {
                case 401 | 403 => PineconeError.authenticationError(s"Authentication error: $bodyStr")
                case 404 => PineconeError.indexNotFoundError(config.indexName)
                case 429 => PineconeError.rateLimitError(s"Rate limit exceeded: $bodyStr")
                case code if code >= 500 => PineconeError.serverError(s"Pinecone server error: $bodyStr")
                case _ => PineconeError.invalidRequestError(s"Invalid request: $bodyStr")
              }
              ZIO.fail(error)
            }
        }
    } yield result
  }

  /**
   * Adds documents to the Pinecone index.
   *
   * @param documents The documents to add
   * @return A ZIO effect that completes when the documents are added
   */
  def addDocuments(documents: Seq[Document]): ZIO[Any, RetrieverError, Unit] =
    if (documents.isEmpty) ZIO.unit
    else
      for {
        // Generate embeddings for the documents
        docEmbeddings <- embeddingModel.embedDocuments(documents)
          .mapError(err => RetrieverError(err, "Failed to generate embeddings for documents"))
        
        // Validate dimensions match the configured dimension
        _ <- ZIO.foreach(docEmbeddings) { case (_, embedding) =>
          if (embedding.values.length != config.dimension)
            ZIO.fail(PineconeError.dimensionMismatchError(config.dimension, embedding.values.length))
          else
            ZIO.unit
        }
        
        // Convert to Pinecone vectors
        vectors = docEmbeddings.map { case (doc, embedding) =>
          Vector(
            id = doc.id,
            values = embedding.values.toArray,
            metadata = Some(
              Metadata(
                text = doc.content,
                metadata = doc.metadata
              )
            )
          )
        }
        
        // Split into batches of 100 (Pinecone limit)
        batches = vectors.grouped(100).toSeq
        
        // Upsert each batch
        _ <- ZIO.foreachDiscard(batches) { batch =>
          val upsertRequest = UpsertRequest(
            vectors = batch.toSeq,
            namespace = config.namespace
          )
          
          makeRequest[UpsertResponse](
            path = "/vectors/upsert",
            method = Method.POST,
            jsonBody = Some(upsertRequest.toJson)
          ).map(_ => ())
        }
      } yield ()

  /**
   * Adds a single document to the Pinecone index.
   *
   * @param document The document to add
   * @return A ZIO effect that completes when the document is added
   */
  def addDocument(document: Document): ZIO[Any, RetrieverError, Unit] =
    addDocuments(Seq(document))

  /**
   * Deletes documents from the Pinecone index by their IDs.
   *
   * @param ids The IDs of the documents to delete
   * @return A ZIO effect that completes when the documents are deleted
   */
  def deleteDocuments(ids: Seq[String]): ZIO[Any, RetrieverError, Unit] =
    if (ids.isEmpty) ZIO.unit
    else
      val deleteRequest = DeleteRequest(
        ids = Some(ids),
        deleteAll = None,
        namespace = config.namespace
      )
      
      makeRequest[DeleteResponse](
        path = "/vectors/delete",
        method = Method.POST,
        jsonBody = Some(deleteRequest.toJson)
      ).map(_ => ())

  /**
   * Deletes all documents from the Pinecone index.
   *
   * @return A ZIO effect that completes when all documents are deleted
   */
  def deleteAll(): ZIO[Any, RetrieverError, Unit] =
    val deleteRequest = DeleteRequest(
      ids = None,
      deleteAll = Some(true),
      namespace = config.namespace
    )
    
    makeRequest[DeleteResponse](
      path = "/vectors/delete",
      method = Method.POST,
      jsonBody = Some(deleteRequest.toJson)
    ).map(_ => ())

  /**
   * Retrieves documents relevant to a query.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of documents
   */
  override def retrieve(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[Document]] =
    retrieveWithScores(query, maxResults).map(_.map(_._1))

  /**
   * Retrieves documents relevant to a query with their similarity scores.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of document-score pairs
   */
  override def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
    for {
      // Generate embedding for the query
      queryEmbedding <- embeddingModel.embedQuery(query)
        .mapError(err => RetrieverError(err, "Failed to generate embedding for query"))
      
      // Validate dimension
      _ <- ZIO.when(queryEmbedding.values.length != config.dimension) {
        ZIO.fail(PineconeError.dimensionMismatchError(config.dimension, queryEmbedding.values.length))
      }
      
      // Create query request
      queryRequest = QueryRequest(
        vector = queryEmbedding.values.toArray,
        topK = maxResults,
        includeMetadata = true,
        includeValues = false,
        namespace = config.namespace
      )
      
      // Send query request
      response <- makeRequest[QueryResponse](
        path = "/query",
        method = Method.POST,
        jsonBody = Some(queryRequest.toJson)
      )
      
      // Convert to documents with scores
      documents = response.matches.flatMap { m =>
        m.metadata.map { metadata =>
          val doc = Document(
            id = m.id,
            content = metadata.text,
            metadata = metadata.metadata
          )
          (doc, m.score.toDouble)
        }
      }
    } yield documents

/**
 * Companion object for PineconeStore.
 */
object PineconeStore:
  /**
   * Creates a PineconeStore.
   *
   * @param config The Pinecone configuration
   * @param embeddingModel The embedding model to use
   * @param client The HTTP client to use for API requests
   * @return A ZIO effect that produces a PineconeStore
   */
  def make(
    config: PineconeConfig,
    embeddingModel: EmbeddingModel,
    client: Client
  ): UIO[PineconeStore] =
    ZIO.succeed(new PineconeStore(config, embeddingModel, client))

  /**
   * Creates a ZLayer that provides a Retriever implementation using Pinecone.
   *
   * @return A ZLayer that requires a PineconeConfig, an EmbeddingModel, and a Client, and provides a Retriever
   */
  val live: ZLayer[PineconeConfig & EmbeddingModel & Client, Nothing, Retriever] =
    ZLayer.fromZIO(
      for {
        config <- ZIO.service[PineconeConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        client <- ZIO.service[Client]
        store <- make(config, embeddingModel, client)
      } yield store
    )

  /**
   * Creates a ZLayer that provides a PineconeStore.
   *
   * @return A ZLayer that requires a PineconeConfig, an EmbeddingModel, and a Client, and provides a PineconeStore
   */
  val liveStore: ZLayer[PineconeConfig & EmbeddingModel & Client, Nothing, PineconeStore] =
    ZLayer.fromZIO(
      for {
        config <- ZIO.service[PineconeConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        client <- ZIO.service[Client]
        store <- make(config, embeddingModel, client)
      } yield store
    )

  /**
   * Creates a scoped ZLayer that provides a PineconeStore.
   * This ensures proper resource cleanup when the scope ends.
   *
   * @return A ZLayer that requires a PineconeConfig, an EmbeddingModel, and a Client, and provides a PineconeStore
   */
  val scoped: ZLayer[PineconeConfig & EmbeddingModel & Client, Nothing, PineconeStore] =
    ZLayer.scoped {
      for {
        config <- ZIO.service[PineconeConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        client <- ZIO.service[Client]
        store <- ZIO.acquireRelease(
          make(config, embeddingModel, client)
        )(_ => ZIO.unit) // No specific cleanup needed for PineconeStore
      } yield store
    }
    
  /**
   * Creates a ZLayer that provides a PineconeStore with a default HTTP client.
   * This is used in the PineconeExample.
   */
  val liveStoreWithDefaultClient: ZLayer[PineconeConfig & EmbeddingModel & Client, Nothing, PineconeStore] = 
    liveStore

/**
 * Internal API models for Pinecone API.
 */
private object PineconeApi:
  /**
   * Metadata for a vector.
   *
   * @param text The text content of the document
   * @param metadata Additional metadata associated with the document
   */
  case class Metadata(
    text: String,
    metadata: Map[String, String]
  )

  object Metadata:
    given JsonEncoder[Metadata] = DeriveJsonEncoder.gen[Metadata]
    given JsonDecoder[Metadata] = DeriveJsonDecoder.gen[Metadata]

  /**
   * A vector in the Pinecone index.
   *
   * @param id The vector ID
   * @param values The vector values
   * @param metadata Optional metadata associated with the vector
   */
  case class Vector(
    id: String,
    values: Array[Float],
    metadata: Option[Metadata] = None
  )

  object Vector:
    given JsonEncoder[Vector] = DeriveJsonEncoder.gen[Vector]
    given JsonDecoder[Vector] = DeriveJsonDecoder.gen[Vector]

  /**
   * Request to upsert vectors to the Pinecone index.
   *
   * @param vectors The vectors to upsert
   * @param namespace Optional namespace
   */
  case class UpsertRequest(
    vectors: Seq[Vector],
    namespace: Option[String] = None
  )

  object UpsertRequest:
    given JsonEncoder[UpsertRequest] = DeriveJsonEncoder.gen[UpsertRequest]

  /**
   * Response from an upsert operation.
   *
   * @param upsertedCount The number of vectors upserted
   */
  case class UpsertResponse(
    upsertedCount: Int
  )

  object UpsertResponse:
    given JsonDecoder[UpsertResponse] = DeriveJsonDecoder.gen[UpsertResponse]
    
  /**
   * Response from a delete operation.
   */
  case class DeleteResponse(
    deleteCount: Int
  )
  
  object DeleteResponse:
    given JsonDecoder[DeleteResponse] = DeriveJsonDecoder.gen[DeleteResponse]

  /**
   * Request to delete vectors from the Pinecone index.
   *
   * @param ids Optional list of vector IDs to delete
   * @param deleteAll Optional flag to delete all vectors
   * @param namespace Optional namespace
   */
  case class DeleteRequest(
    ids: Option[Seq[String]] = None,
    deleteAll: Option[Boolean] = None,
    namespace: Option[String] = None
  )

  object DeleteRequest:
    given JsonEncoder[DeleteRequest] = DeriveJsonEncoder.gen[DeleteRequest]

  /**
   * Request to query the Pinecone index.
   *
   * @param vector The query vector
   * @param topK The number of results to return
   * @param includeMetadata Whether to include metadata in the response
   * @param includeValues Whether to include vector values in the response
   * @param namespace Optional namespace
   */
  case class QueryRequest(
    vector: Array[Float],
    topK: Int,
    includeMetadata: Boolean,
    includeValues: Boolean,
    namespace: Option[String] = None
  )

  object QueryRequest:
    given JsonEncoder[QueryRequest] = DeriveJsonEncoder.gen[QueryRequest]

  /**
   * A match from a query operation.
   *
   * @param id The vector ID
   * @param score The similarity score
   * @param metadata Optional metadata associated with the vector
   * @param values Optional vector values
   */
  case class Match(
    id: String,
    score: Float,
    metadata: Option[Metadata] = None,
    values: Option[Array[Float]] = None
  )

  object Match:
    given JsonDecoder[Match] = DeriveJsonDecoder.gen[Match]

  /**
   * Response from a query operation.
   *
   * @param matches The matching vectors
   * @param namespace The namespace of the query
   */
  case class QueryResponse(
    matches: Seq[Match],
    namespace: String
  )

  object QueryResponse:
    given JsonDecoder[QueryResponse] = DeriveJsonDecoder.gen[QueryResponse]