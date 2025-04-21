package zio.langchain.rag

import zio.*
import zio.langchain.core.model.*
import zio.langchain.core.errors.*

/**
 * Error type for query transformation operations.
 *
 * @param cause The underlying cause of the error
 * @param message A descriptive error message
 */
case class QueryTransformationError(
  cause: Throwable,
  message: String = "Query transformation error occurred"
) extends LangChainError:
  override def getMessage: String = s"$message: ${cause.getMessage}"
  override def getCause: Throwable = cause

/**
 * Interface for query transformers.
 * Provides methods for transforming user queries to improve retrieval performance.
 */
trait QueryTransformer:
  /**
   * Transforms a query to improve retrieval performance.
   *
   * @param query The original query string
   * @return A ZIO effect that produces a transformed query string or fails with a QueryTransformationError
   */
  def transform(query: String): ZIO[Any, QueryTransformationError, String]

/**
 * Companion object for QueryTransformer.
 */
object QueryTransformer:
  /**
   * Creates a ZIO accessor for the QueryTransformer service.
   *
   * @return A ZIO effect that requires a QueryTransformer and produces the QueryTransformer
   */
  def get: ZIO[QueryTransformer, Nothing, QueryTransformer] = ZIO.service[QueryTransformer]
  
  /**
   * Transforms a query using the QueryTransformer service.
   *
   * @param query The original query string
   * @return A ZIO effect that requires a QueryTransformer and produces a transformed query string or fails with a QueryTransformationError
   */
  def transform(query: String): ZIO[QueryTransformer, QueryTransformationError, String] =
    ZIO.serviceWithZIO[QueryTransformer](_.transform(query))

  /**
   * Creates a QueryTransformer that applies a sequence of transformations in order.
   *
   * @param transformers The sequence of transformers to apply
   * @return A new QueryTransformer that applies all transformations in sequence
   */
  def compose(transformers: QueryTransformer*): QueryTransformer = new QueryTransformer:
    override def transform(query: String): ZIO[Any, QueryTransformationError, String] =
      transformers.foldLeft(ZIO.succeed(query)) { (queryEffect, transformer) =>
        queryEffect.flatMap(q => transformer.transform(q).orDie)
      }

  /**
   * Creates an identity QueryTransformer that returns the original query unchanged.
   *
   * @return A QueryTransformer that doesn't modify the query
   */
  def identity: QueryTransformer = new QueryTransformer:
    override def transform(query: String): ZIO[Any, QueryTransformationError, String] =
      ZIO.succeed(query)

/**
 * LLM-powered query expansion transformer.
 * Expands the original query to include more relevant terms and context.
 *
 * @param llm The LLM service to use for query expansion
 */
class QueryExpansionTransformer(llm: LLM) extends QueryTransformer:
  override def transform(query: String): ZIO[Any, QueryTransformationError, String] =
    val prompt = s"""You are an AI assistant helping to improve search queries. 
                    |Your task is to expand the given query to make it more effective for retrieval.
                    |Add relevant terms, synonyms, or context that would help in finding the most relevant information.
                    |Keep the expanded query concise and focused on the original intent.
                    |
                    |Original query: "$query"
                    |
                    |Expanded query:""".stripMargin
    
    llm.complete(prompt)
      .mapError(e => QueryTransformationError(e, s"Failed to expand query: ${e.getMessage}"))

/**
 * Companion object for QueryExpansionTransformer.
 */
object QueryExpansionTransformer:
  /**
   * Creates a live layer for the QueryExpansionTransformer.
   *
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  val live: ZLayer[LLM, Nothing, QueryTransformer] =
    ZLayer.fromFunction(llm => new QueryExpansionTransformer(llm))

/**
 * Hypothetical Document Embeddings (HyDE) transformer.
 * Generates a hypothetical document that would answer the query, then uses that document
 * as the query for retrieval.
 *
 * @param llm The LLM service to use for generating hypothetical documents
 */
class HyDETransformer(llm: LLM) extends QueryTransformer:
  override def transform(query: String): ZIO[Any, QueryTransformationError, String] =
    val prompt = s"""You are an AI assistant helping to improve search queries using the Hypothetical Document Embeddings technique.
                    |Your task is to generate a hypothetical document or passage that would be the perfect answer to the given query.
                    |This hypothetical document will be used as the query for retrieval instead of the original question.
                    |
                    |Query: "$query"
                    |
                    |Hypothetical document that would answer this query:""".stripMargin
    
    llm.complete(prompt)
      .mapError(e => QueryTransformationError(e, s"Failed to generate hypothetical document: ${e.getMessage}"))

/**
 * Companion object for HyDETransformer.
 */
object HyDETransformer:
  /**
   * Creates a live layer for the HyDETransformer.
   *
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  val live: ZLayer[LLM, Nothing, QueryTransformer] =
    ZLayer.fromFunction(llm => new HyDETransformer(llm))

/**
 * Multi-query transformer that generates multiple search queries from a single user question.
 * This helps capture different aspects or interpretations of the original query.
 *
 * @param llm The LLM service to use for generating multiple queries
 * @param numQueries The number of queries to generate (default: 3)
 */
class MultiQueryTransformer(llm: LLM, numQueries: Int = 3) extends QueryTransformer:
  override def transform(query: String): ZIO[Any, QueryTransformationError, String] =
    val prompt = s"""You are an AI assistant helping to improve search queries.
                    |Your task is to generate $numQueries different versions of the given query.
                    |Each version should focus on a different aspect or use different terms while preserving the original intent.
                    |Format your response as a single string with queries separated by newlines.
                    |
                    |Original query: "$query"
                    |
                    |$numQueries alternative queries:""".stripMargin
    
    llm.complete(prompt)
      .map(result => s"$query\n$result") // Include the original query along with alternatives
      .mapError(e => QueryTransformationError(e, s"Failed to generate multiple queries: ${e.getMessage}"))

/**
 * Companion object for MultiQueryTransformer.
 */
object MultiQueryTransformer:
  /**
   * Creates a live layer for the MultiQueryTransformer.
   *
   * @param numQueries The number of queries to generate (default: 3)
   * @return A ZLayer that requires an LLM and produces a QueryTransformer
   */
  def live(numQueries: Int = 3): ZLayer[LLM, Nothing, QueryTransformer] =
    ZLayer.fromFunction(llm => new MultiQueryTransformer(llm, numQueries))