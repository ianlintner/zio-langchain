package zio.langchain.core.model

import zio.*

import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * Interface for embedding models.
 * Provides methods for generating vector embeddings from text.
 */
trait EmbeddingModel:
  /**
   * Generates an embedding for a single text.
   *
   * @param text The text to embed
   * @return A ZIO effect that produces an Embedding or fails with an EmbeddingError
   */
  def embed(text: String): ZIO[Any, EmbeddingError, Embedding]
  
  /**
   * Generates an embedding for a query.
   * This is an alias for embed that makes the intent clearer when used for retrieval.
   *
   * @param query The query text to embed
   * @return A ZIO effect that produces an Embedding or fails with an EmbeddingError
   */
  def embedQuery(query: String): ZIO[Any, EmbeddingError, Embedding] =
    embed(query)
  
  /**
   * Generates embeddings for multiple texts.
   *
   * @param texts The sequence of texts to embed
   * @return A ZIO effect that produces a sequence of Embeddings or fails with an EmbeddingError
   */
  def embedAll(texts: Seq[String]): ZIO[Any, EmbeddingError, Seq[Embedding]]
  
  /**
   * Generates an embedding for a document.
   *
   * @param document The document to embed
   * @return A ZIO effect that produces a tuple of the document and its embedding, or fails with an EmbeddingError
   */
  def embedDocument(document: Document): ZIO[Any, EmbeddingError, (Document, Embedding)] =
    embed(document.content).map(embedding => (document, embedding))
    
  /**
   * Generates embeddings for multiple documents.
   *
   * @param documents The sequence of documents to embed
   * @return A ZIO effect that produces a sequence of document-embedding pairs, or fails with an EmbeddingError
   */
  def embedDocuments(documents: Seq[Document]): ZIO[Any, EmbeddingError, Seq[(Document, Embedding)]] =
    ZIO.foreachPar(documents)(embedDocument)

/**
 * Companion object for EmbeddingModel.
 */
object EmbeddingModel:
  /**
   * Creates a ZIO accessor for the EmbeddingModel service.
   *
   * @return A ZIO effect that requires an EmbeddingModel and produces the EmbeddingModel
   */
  def get: ZIO[EmbeddingModel, Nothing, EmbeddingModel] = ZIO.service[EmbeddingModel]
  
  /**
   * Generates an embedding for a single text using the EmbeddingModel service.
   *
   * @param text The text to embed
   * @return A ZIO effect that requires an EmbeddingModel and produces an Embedding or fails with an EmbeddingError
   */
  def embed(text: String): ZIO[EmbeddingModel, EmbeddingError, Embedding] =
    ZIO.serviceWithZIO[EmbeddingModel](_.embed(text))
  
  /**
   * Generates embeddings for multiple texts using the EmbeddingModel service.
   *
   * @param texts The sequence of texts to embed
   * @return A ZIO effect that requires an EmbeddingModel and produces a sequence of Embeddings or fails with an EmbeddingError
   */
  def embedAll(texts: Seq[String]): ZIO[EmbeddingModel, EmbeddingError, Seq[Embedding]] =
    ZIO.serviceWithZIO[EmbeddingModel](_.embedAll(texts))
    
  /**
   * Generates an embedding for a query using the EmbeddingModel service.
   *
   * @param query The query text to embed
   * @return A ZIO effect that requires an EmbeddingModel and produces an Embedding or fails with an EmbeddingError
   */
  def embedQuery(query: String): ZIO[EmbeddingModel, EmbeddingError, Embedding] =
    ZIO.serviceWithZIO[EmbeddingModel](_.embedQuery(query))
  
  /**
   * Generates an embedding for a document using the EmbeddingModel service.
   *
   * @param document The document to embed
   * @return A ZIO effect that requires an EmbeddingModel and produces a tuple of the document and its embedding, or fails with an EmbeddingError
   */
  def embedDocument(document: Document): ZIO[EmbeddingModel, EmbeddingError, (Document, Embedding)] =
    ZIO.serviceWithZIO[EmbeddingModel](_.embedDocument(document))
    
  /**
   * Generates embeddings for multiple documents using the EmbeddingModel service.
   *
   * @param documents The sequence of documents to embed
   * @return A ZIO effect that requires an EmbeddingModel and produces a sequence of document-embedding pairs, or fails with an EmbeddingError
   */
  def embedDocuments(documents: Seq[Document]): ZIO[EmbeddingModel, EmbeddingError, Seq[(Document, Embedding)]] =
    ZIO.serviceWithZIO[EmbeddingModel](_.embedDocuments(documents))