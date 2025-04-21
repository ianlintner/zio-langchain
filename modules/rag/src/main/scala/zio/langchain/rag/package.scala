package zio.langchain

import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.retriever.Retriever
import zio.langchain.rag.{QueryTransformer, TransformingRetriever, MultiTransformingRetriever}

/**
 * The rag package provides advanced Retrieval-Augmented Generation (RAG) capabilities,
 * including query transformation techniques to improve retrieval performance.
 */
package object rag:
  /**
   * Creates a query expansion transformer.
   *
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  val queryExpansionTransformer: ZLayer[LLM, Nothing, QueryTransformer] =
    QueryExpansionTransformer.live
  
  /**
   * Creates a Hypothetical Document Embeddings (HyDE) transformer.
   *
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  val hydeTransformer: ZLayer[LLM, Nothing, QueryTransformer] =
    HyDETransformer.live
  
  /**
   * Creates a multi-query transformer.
   *
   * @param numQueries The number of queries to generate (default: 3)
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  def multiQueryTransformer(numQueries: Int = 3): ZLayer[LLM, Nothing, QueryTransformer] =
    MultiQueryTransformer.live(numQueries)
  
  /**
   * Creates a transforming retriever that applies query transformation before retrieval.
   *
   * @return A ZLayer that requires a Retriever and a QueryTransformer and produces a Retriever
   */
  val transformingRetriever: ZLayer[Retriever & QueryTransformer, Nothing, Retriever] =
    TransformingRetriever.live
  
  /**
   * Creates a composite query transformer that applies multiple transformations in sequence.
   *
   * @param transformers The transformers to compose
   * @return A QueryTransformer that applies all transformations in sequence
   */
  def composeTransformers(transformers: QueryTransformer*): QueryTransformer =
    QueryTransformer.compose(transformers*)
  
  /**
   * Creates a multi-transforming retriever that applies multiple query transformations
   * and merges the results.
   *
   * @param retriever The underlying retriever to use
   * @param transformers The query transformers to apply
   * @param deduplicateResults Whether to deduplicate results by document ID (default: true)
   * @return A new MultiTransformingRetriever instance
   */
  def createMultiTransformingRetriever(
    retriever: Retriever,
    transformers: Seq[QueryTransformer],
    deduplicateResults: Boolean = true
  ): MultiTransformingRetriever =
    MultiTransformingRetriever(retriever, transformers, deduplicateResults)