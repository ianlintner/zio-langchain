package zio.langchain.core.config

import zio.*
import zio.config.*
import zio.config.typesafe.*

import java.time.Duration

/**
 * Base trait for all configuration in ZIO LangChain.
 */
trait LangChainConfig

/**
 * Base trait for model configurations.
 */
trait ModelConfig extends LangChainConfig:
  /**
   * The model identifier or name.
   *
   * @return The model identifier
   */
  def model: String
  
  /**
   * The temperature parameter for the model.
   * Higher values make the output more random, lower values make it more deterministic.
   *
   * @return The temperature value
   */
  def temperature: Double
  
  /**
   * The maximum number of tokens to generate.
   *
   * @return The maximum number of tokens, if specified
   */
  def maxTokens: Option[Int]
  
  /**
   * The timeout for model requests.
   *
   * @return The timeout duration
   */
  def timeout: Duration

/**
 * Base trait for embedding model configurations.
 */
trait EmbeddingConfig extends LangChainConfig:
  /**
   * The model identifier or name.
   *
   * @return The model identifier
   */
  def model: String
  
  /**
   * The timeout for embedding requests.
   *
   * @return The timeout duration
   */
  def timeout: Duration

/**
 * Base trait for retriever configurations.
 */
trait RetrieverConfig extends LangChainConfig:
  /**
   * The default number of results to return.
   *
   * @return The default number of results
   */
  def defaultMaxResults: Int

/**
 * Simplified configuration utilities that will be expanded later
 * when the project's use of ZIO Config is better established.
 */
object config:
  /**
   * Creates a ZLayer that contains the provided configuration value.
   *
   * @param value The configuration value
   * @return A ZLayer that provides the configuration
   */
  def layer[A: Tag](value: A): ZLayer[Any, Nothing, A] =
    ZLayer.succeed(value)