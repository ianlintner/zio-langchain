package zio.langchain.core.retriever

import zio.*

import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * Interface for document retrievers.
 * Provides methods for retrieving relevant documents based on a query.
 */
trait Retriever:
  /**
   * Retrieves documents relevant to a query.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return (default: 10)
   * @return A ZIO effect that produces a sequence of documents or fails with a RetrieverError
   */
  def retrieve(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[Document]]
  
  /**
   * Retrieves documents relevant to a query with their similarity scores.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return (default: 10)
   * @return A ZIO effect that produces a sequence of document-score pairs or fails with a RetrieverError
   */
  def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
    retrieve(query, maxResults).map(_.map((_, 1.0)))

/**
 * Companion object for Retriever.
 */
object Retriever:
  /**
   * Creates a ZIO accessor for the Retriever service.
   *
   * @return A ZIO effect that requires a Retriever and produces the Retriever
   */
  def get: ZIO[Retriever, Nothing, Retriever] = ZIO.service[Retriever]
  
  /**
   * Retrieves documents relevant to a query using the Retriever service.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return (default: 10)
   * @return A ZIO effect that requires a Retriever and produces a sequence of documents or fails with a RetrieverError
   */
  def retrieve(query: String, maxResults: Int = 10): ZIO[Retriever, RetrieverError, Seq[Document]] =
    ZIO.serviceWithZIO[Retriever](_.retrieve(query, maxResults))
  
  /**
   * Retrieves documents relevant to a query with their similarity scores using the Retriever service.
   *
   * @param query The query string
   * @param maxResults The maximum number of results to return (default: 10)
   * @return A ZIO effect that requires a Retriever and produces a sequence of document-score pairs or fails with a RetrieverError
   */
  def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Retriever, RetrieverError, Seq[(Document, Double)]] =
    ZIO.serviceWithZIO[Retriever](_.retrieveWithScores(query, maxResults))