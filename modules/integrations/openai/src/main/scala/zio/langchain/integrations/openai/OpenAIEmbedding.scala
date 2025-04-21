package zio.langchain.integrations.openai

import zio.*

import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.embedding.EmbeddingModel as LC4JEmbeddingModel

import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

import scala.jdk.CollectionConverters.*
import java.util.concurrent.TimeUnit

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
  timeout: java.time.Duration = java.time.Duration.ofSeconds(60)
) extends zio.langchain.core.config.EmbeddingConfig

/**
 * Companion object for OpenAIEmbeddingConfig.
 */
object OpenAIEmbeddingConfig:
  /**
   * Automatically derived ConfigDescriptor for OpenAIEmbeddingConfig.
   */
  given zio.config.ConfigDescriptor[OpenAIEmbeddingConfig] = zio.config.magnolia.descriptor[OpenAIEmbeddingConfig]
  
  /**
   * Loads OpenAIEmbeddingConfig from the "openai.embedding" section of the configuration.
   *
   * @return A ZIO effect that produces an OpenAIEmbeddingConfig or fails with a ConfigError
   */
  val load: ZIO[Any, zio.config.ConfigError, OpenAIEmbeddingConfig] =
    zio.langchain.core.config.config.loadAt[OpenAIEmbeddingConfig]("openai.embedding")
  
  /**
   * Creates a ZLayer from the "openai.embedding" section of the configuration.
   *
   * @return A ZLayer that provides an OpenAIEmbeddingConfig
   */
  val layer: ZLayer[Any, zio.config.ConfigError, OpenAIEmbeddingConfig] =
    ZLayer.fromZIO(load)

/**
 * Implementation of the EmbeddingModel interface for OpenAI models.
 *
 * @param client The langchain4j OpenAI embedding client
 * @param config The OpenAI embedding configuration
 */
class OpenAIEmbedding(
  client: LC4JEmbeddingModel,
  config: OpenAIEmbeddingConfig
) extends EmbeddingModel:
  /**
   * Generates an embedding for a single text.
   *
   * @param text The text to embed
   * @return A ZIO effect that produces an Embedding or fails with an EmbeddingError
   */
  override def embed(text: String): ZIO[Any, EmbeddingError, Embedding] =
    ZIO.attemptBlockingIO {
      val embedding = client.embed(text)
      Embedding(embedding.vectorAsList.asScala.map(_.toFloat).toVector)
    }.mapError(e => EmbeddingError(e))
  
  /**
   * Generates embeddings for multiple texts.
   *
   * @param texts The sequence of texts to embed
   * @return A ZIO effect that produces a sequence of Embeddings or fails with an EmbeddingError
   */
  override def embedAll(texts: Seq[String]): ZIO[Any, EmbeddingError, Seq[Embedding]] =
    ZIO.attemptBlockingIO {
      val embeddings = client.embedAll(texts.asJava)
      embeddings.asScala.map(e => 
        Embedding(e.vectorAsList.asScala.map(_.toFloat).toVector)
      ).toSeq
    }.mapError(e => EmbeddingError(e))

/**
 * Companion object for OpenAIEmbedding.
 */
object OpenAIEmbedding:
  /**
   * Creates an OpenAIEmbedding from an OpenAIEmbeddingConfig.
   *
   * @param config The OpenAI embedding configuration
   * @return A ZIO effect that produces an OpenAIEmbedding or fails with a Throwable
   */
  def make(config: OpenAIEmbeddingConfig): ZIO[Any, Throwable, OpenAIEmbedding] =
    ZIO.attempt {
      val client = OpenAiEmbeddingModel.builder()
        .apiKey(config.apiKey)
        .modelName(config.model)
        .organizationId(config.organizationId.orNull)
        .timeout(config.timeout.toMillis, TimeUnit.MILLISECONDS)
        .build()
      
      new OpenAIEmbedding(client, config)
    }
  
  /**
   * Creates a ZLayer that provides an EmbeddingModel implementation using OpenAI.
   *
   * @return A ZLayer that requires an OpenAIEmbeddingConfig and provides an EmbeddingModel
   */
  val live: ZLayer[OpenAIEmbeddingConfig, Throwable, EmbeddingModel] =
    ZLayer {
      for
        config <- ZIO.service[OpenAIEmbeddingConfig]
        embedding <- make(config)
      yield embedding
    }