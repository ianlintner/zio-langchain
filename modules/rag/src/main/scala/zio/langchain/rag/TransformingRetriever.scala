package zio.langchain.rag

import zio.*
import zio.langchain.core.retriever.*
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * A retriever that transforms queries before passing them to an underlying retriever.
 * This allows for improved retrieval performance through query transformation techniques.
 *
 * @param underlying The underlying retriever to use for document retrieval
 * @param transformer The query transformer to apply before retrieval
 */
class TransformingRetriever(
  underlying: Retriever,
  transformer: QueryTransformer
) extends Retriever:
  /**
   * Retrieves documents relevant to a query after applying query transformation.
   *
   * @param query The original query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of documents or fails with a RetrieverError
   */
  override def retrieve(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[Document]] =
    transformer.transform(query)
      .mapError(e => RetrieverError(e, s"Query transformation failed: ${e.getMessage}"))
      .flatMap(transformedQuery => 
        ZIO.logInfo(s"Transformed query: '$query' -> '$transformedQuery'") *>
        underlying.retrieve(transformedQuery, maxResults)
      )
  
  /**
   * Retrieves documents relevant to a query with their similarity scores after applying query transformation.
   *
   * @param query The original query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of document-score pairs or fails with a RetrieverError
   */
  override def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
    transformer.transform(query)
      .mapError(e => RetrieverError(e, s"Query transformation failed: ${e.getMessage}"))
      .flatMap(transformedQuery => 
        ZIO.logInfo(s"Transformed query: '$query' -> '$transformedQuery'") *>
        underlying.retrieveWithScores(transformedQuery, maxResults)
      )

/**
 * Companion object for TransformingRetriever.
 */
object TransformingRetriever:
  /**
   * Creates a live layer for the TransformingRetriever.
   *
   * @return A ZLayer that requires a Retriever and a QueryTransformer and produces a Retriever
   */
  val live: ZLayer[Retriever & QueryTransformer, Nothing, Retriever] =
    ZLayer.fromFunction((retriever: Retriever, transformer: QueryTransformer) => 
      new TransformingRetriever(retriever, transformer)
    )
  
  /**
   * Creates a TransformingRetriever with the specified underlying retriever and transformer.
   *
   * @param retriever The underlying retriever to use
   * @param transformer The query transformer to apply
   * @return A new TransformingRetriever instance
   */
  def apply(retriever: Retriever, transformer: QueryTransformer): TransformingRetriever =
    new TransformingRetriever(retriever, transformer)

/**
 * A retriever that applies multiple query transformations and merges the results.
 * This is useful for techniques like multi-query retrieval where different transformations
 * of the same query are used to retrieve a diverse set of documents.
 *
 * @param underlying The underlying retriever to use for document retrieval
 * @param transformers The query transformers to apply
 * @param deduplicateResults Whether to deduplicate results by document ID (default: true)
 */
class MultiTransformingRetriever(
  underlying: Retriever,
  transformers: Seq[QueryTransformer],
  deduplicateResults: Boolean = true
) extends Retriever:
  /**
   * Retrieves documents by applying multiple query transformations and merging the results.
   *
   * @param query The original query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of documents or fails with a RetrieverError
   */
  override def retrieve(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[Document]] =
    // Apply each transformer and retrieve documents
    ZIO.foreach(transformers) { transformer =>
      transformer.transform(query)
        .mapError(e => RetrieverError(e, s"Query transformation failed: ${e.getMessage}"))
        .tap(transformedQuery => ZIO.logInfo(s"Transformed query: '$query' -> '$transformedQuery'"))
        .flatMap(transformedQuery => underlying.retrieve(transformedQuery, maxResults))
    }.map { results =>
      // Flatten and deduplicate results if needed
      val allDocs = results.flatten
      if deduplicateResults then
        allDocs.distinctBy(_.id).take(maxResults)
      else
        allDocs.take(maxResults)
    }
  
  /**
   * Retrieves documents with scores by applying multiple query transformations and merging the results.
   *
   * @param query The original query string
   * @param maxResults The maximum number of results to return
   * @return A ZIO effect that produces a sequence of document-score pairs or fails with a RetrieverError
   */
  override def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
    // Apply each transformer and retrieve documents with scores
    ZIO.foreach(transformers) { transformer =>
      transformer.transform(query)
        .mapError(e => RetrieverError(e, s"Query transformation failed: ${e.getMessage}"))
        .tap(transformedQuery => ZIO.logInfo(s"Transformed query: '$query' -> '$transformedQuery'"))
        .flatMap(transformedQuery => underlying.retrieveWithScores(transformedQuery, maxResults))
    }.map { results =>
      // Flatten and deduplicate results if needed
      val allDocsWithScores = results.flatten
      if deduplicateResults then
        // When duplicates exist, keep the one with the highest score
        allDocsWithScores
          .groupBy(_._1.id)
          .map { case (_, docs) => docs.maxBy(_._2) }
          .toSeq
          .sortBy(-_._2)
          .take(maxResults)
      else
        allDocsWithScores
          .sortBy(-_._2)
          .take(maxResults)
    }

/**
 * Companion object for MultiTransformingRetriever.
 */
object MultiTransformingRetriever:
  /**
   * Creates a MultiTransformingRetriever with the specified underlying retriever and transformers.
   *
   * @param retriever The underlying retriever to use
   * @param transformers The query transformers to apply
   * @param deduplicateResults Whether to deduplicate results by document ID (default: true)
   * @return A new MultiTransformingRetriever instance
   */
  def apply(
    retriever: Retriever, 
    transformers: Seq[QueryTransformer],
    deduplicateResults: Boolean = true
  ): MultiTransformingRetriever =
    new MultiTransformingRetriever(retriever, transformers, deduplicateResults)