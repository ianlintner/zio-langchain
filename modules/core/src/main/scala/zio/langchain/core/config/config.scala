package zio.langchain.core.config

import zio.*
import zio.config.*
import zio.config.magnolia.descriptor
import zio.config.typesafe.TypesafeConfigProvider

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
 * Utilities for working with configurations.
 */
object config:
  /**
   * Loads a configuration of the specified type from the default configuration sources.
   * This includes environment variables, system properties, and application.conf.
   *
   * @tparam A The configuration type to load
   * @return A ZIO effect that produces the configuration or fails with a ConfigError
   */
  inline def load[A: Tag](using ConfigDescriptor[A]): ZIO[Any, ConfigError, A] =
    ZIO.config[A].provideLayer(
      TypesafeConfigProvider.fromResourcePath().orElse(
        TypesafeConfigProvider.fromDefaultLoader()
      )
    )
  
  /**
   * Loads a configuration of the specified type from a specific path in the configuration.
   *
   * @param path The path to load the configuration from
   * @tparam A The configuration type to load
   * @return A ZIO effect that produces the configuration or fails with a ConfigError
   */
  inline def loadAt[A: Tag](path: String)(using ConfigDescriptor[A]): ZIO[Any, ConfigError, A] =
    ZIO.config[A](Config.nested(path)).provideLayer(
      TypesafeConfigProvider.fromResourcePath().orElse(
        TypesafeConfigProvider.fromDefaultLoader()
      )
    )
  
  /**
   * Creates a ZLayer from a configuration of the specified type.
   *
   * @tparam A The configuration type to load
   * @return A ZLayer that provides the configuration
   */
  inline def layer[A: Tag](using ConfigDescriptor[A]): ZLayer[Any, ConfigError, A] =
    ZLayer.fromZIO(load[A])
  
  /**
   * Creates a ZLayer from a configuration of the specified type at a specific path.
   *
   * @param path The path to load the configuration from
   * @tparam A The configuration type to load
   * @return A ZLayer that provides the configuration
   */
  inline def layerAt[A: Tag](path: String)(using ConfigDescriptor[A]): ZLayer[Any, ConfigError, A] =
    ZLayer.fromZIO(loadAt[A](path))