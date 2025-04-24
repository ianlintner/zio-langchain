package zio.langchain.integrations.pinecone

import zio.*
import zio.json.*
import zio.http.*
import zio.stream.ZStream

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
 */
class PineconeStore private (
  config: PineconeConfig,
  embeddingModel: EmbeddingModel
) extends Retriever:
  import PineconeApi.*

  /**
   * The base URL for the Pinecone API.
   */
  private val baseUrl = s"https://${config.indexName}-${config.projectId}.svc.${config.environment}.pinecone.io"

  /**
   * Makes an HTTP request to the Pinecone API using ZIO HTTP.
   *
   * @param path The API endpoint path
   * @param method The HTTP method
   * @param jsonBody The request body as a JSON string
   * @return A ZIO effect that produces the response body as a string
   */
  private def makeRequest(
    path: String,
    method: Method,
    jsonBody: Option[String] = None
  ): ZIO[Any, Throwable, String] = {
    // Create headers
    val headers = Headers(
      Header.ContentType(MediaType.application.json),
      Header.Custom("Api-Key", config.apiKey)
    )
    
    // Create request body if provided
    val body = jsonBody match {
      case Some(json) => Body.fromString(json)
      case None => Body.empty
    }
    
    // Create and send request
    ZIO.scoped {
      for {
        // Get client
        client <- ZIO.service[Client].provide(Client.default)
        
        // Parse URL
        url <- ZIO.fromEither(URL.decode(s"$baseUrl$path"))
                .orElseFail(new RuntimeException(s"Invalid URL: $baseUrl$path"))
        
        // Create request
        request = method match {
          case Method.GET => Request.get(url).updateHeaders(_ ++ headers)
          case Method.POST => Request.post(body, url).updateHeaders(_ ++ headers)
          case Method.PUT => Request.put(body, url).updateHeaders(_ ++ headers)
          case Method.DELETE => Request.delete(url).updateHeaders(_ ++ headers)
          case _ => Request.apply(body, Headers.empty, method, url, Version.Http_1_1, None).updateHeaders(_ => headers)
        }
        
        // Send request
        response <- client.request(request)
                     .timeoutFail(new RuntimeException("Request timed out"))(config.timeout)
        
        // Check for errors
        _ <- ZIO.when(response.status.isError) {
          response.body.asString.flatMap { body =>
            ZIO.fail(new RuntimeException(s"Pinecone API error: ${response.status.code}, body: $body"))
          }
        }
        
        // Get response body
        responseBody <- response.body.asString
      } yield responseBody
    }
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
          
          makeRequest(
            path = "/vectors/upsert",
            method = Method.POST,
            jsonBody = Some(upsertRequest.toJson)
          ).mapError(err => RetrieverError(err, "Failed to upsert vectors to Pinecone"))
            .unit
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
      
      makeRequest(
        path = "/vectors/delete",
        method = Method.POST,
        jsonBody = Some(deleteRequest.toJson)
      ).mapError(err => RetrieverError(err, "Failed to delete vectors from Pinecone"))
        .unit

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
    
    makeRequest(
      path = "/vectors/delete",
      method = Method.POST,
      jsonBody = Some(deleteRequest.toJson)
    ).mapError(err => RetrieverError(err, "Failed to delete all vectors from Pinecone"))
      .unit

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
      
      // Create query request
      queryRequest = QueryRequest(
        vector = queryEmbedding.values.toArray,
        topK = maxResults,
        includeMetadata = true,
        includeValues = false,
        namespace = config.namespace
      )
      
      // Send query request
      responseJson <- makeRequest(
        path = "/query",
        method = Method.POST,
        jsonBody = Some(queryRequest.toJson)
      ).mapError(err => RetrieverError(err, "Failed to query Pinecone"))
      
      // Parse response
      response <- ZIO.fromEither(responseJson.fromJson[QueryResponse])
        .mapError(err => RetrieverError(new RuntimeException(s"Failed to parse Pinecone response: $err"), ""))
      
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
   * @return A ZIO effect that produces a PineconeStore
   */
  def make(
    config: PineconeConfig,
    embeddingModel: EmbeddingModel
  ): ZIO[Any, Nothing, PineconeStore] =
    ZIO.succeed(new PineconeStore(config, embeddingModel))

  /**
   * Creates a ZLayer that provides a Retriever implementation using Pinecone.
   *
   * @return A ZLayer that requires a PineconeConfig and an EmbeddingModel and provides a Retriever
   */
  val live: ZLayer[PineconeConfig & EmbeddingModel, Nothing, Retriever] =
    ZLayer {
      for
        config <- ZIO.service[PineconeConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        store <- make(config, embeddingModel)
      yield store
    }

  /**
   * Creates a ZLayer that provides a PineconeStore.
   *
   * @return A ZLayer that requires a PineconeConfig and an EmbeddingModel and provides a PineconeStore
   */
  val liveStore: ZLayer[PineconeConfig & EmbeddingModel, Nothing, PineconeStore] =
    ZLayer {
      for
        config <- ZIO.service[PineconeConfig]
        embeddingModel <- ZIO.service[EmbeddingModel]
        store <- make(config, embeddingModel)
      yield store
    }

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