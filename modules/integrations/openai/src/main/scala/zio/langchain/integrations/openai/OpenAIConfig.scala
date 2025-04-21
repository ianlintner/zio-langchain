package zio.langchain.integrations.openai

import zio.*
import zio.config.*
import zio.config.magnolia.descriptor

import java.time.Duration

import zio.langchain.core.config.ModelConfig

/**
 * Configuration for OpenAI models.
 *
 * @param apiKey The OpenAI API key
 * @param model The model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
 * @param temperature The temperature parameter (0.0 to 1.0)
 * @param maxTokens The maximum number of tokens to generate (optional)
 * @param organizationId The OpenAI organization ID (optional)
 * @param timeout The timeout for API requests
 * @param enableStreaming Whether to enable streaming responses (if supported by the model)
 * @param logRequests Whether to log API requests
 * @param logResponses Whether to log API responses
 */
case class OpenAIConfig(
  apiKey: String,
  model: String,
  temperature: Double = 0.7,
  maxTokens: Option[Int] = None,
  organizationId: Option[String] = None,
  timeout: Duration = Duration.ofSeconds(60),
  enableStreaming: Boolean = true,
  logRequests: Boolean = false,
  logResponses: Boolean = false
) extends ModelConfig

/**
 * Companion object for OpenAIConfig.
 */
object OpenAIConfig:
  /**
   * Automatically derived ConfigDescriptor for OpenAIConfig.
   */
  given ConfigDescriptor[OpenAIConfig] = descriptor[OpenAIConfig]
  
  /**
   * Loads OpenAIConfig from the "openai" section of the configuration.
   *
   * @return A ZIO effect that produces an OpenAIConfig or fails with a ConfigError
   */
  val load: ZIO[Any, ConfigError, OpenAIConfig] =
    zio.langchain.core.config.config.loadAt[OpenAIConfig]("openai")
  
  /**
   * Creates a ZLayer from the "openai" section of the configuration.
   *
   * @return A ZLayer that provides an OpenAIConfig
   */
  val layer: ZLayer[Any, ConfigError, OpenAIConfig] =
    ZLayer.fromZIO(load)