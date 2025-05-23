package zio.langchain.integrations.anthropic

import zio.*
import zio.langchain.core.config.ModelConfig

import java.time.Duration as JDuration

/**
 * Configuration for Anthropic Claude models.
 *
 * @param apiKey The Anthropic API key
 * @param model The model identifier (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
 * @param temperature The temperature parameter (0.0 to 1.0)
 * @param maxTokens The maximum number of tokens to generate (optional)
 * @param timeout The timeout for API requests
 * @param enableStreaming Whether to enable streaming responses (if supported by the model)
 * @param logRequests Whether to log API requests
 * @param logResponses Whether to log API responses
 */
case class AnthropicConfig(
  apiKey: String,
  model: String,
  temperature: Double = 0.7,
  maxTokens: Option[Int] = None,
  timeout: Duration = Duration.fromSeconds(60),
  enableStreaming: Boolean = true,
  logRequests: Boolean = false,
  logResponses: Boolean = false
) extends ModelConfig

/**
 * Companion object for AnthropicConfig.
 */
object AnthropicConfig:
  /**
   * Creates an AnthropicConfig from environment variables.
   */
  private def createFromEnv: AnthropicConfig =
    AnthropicConfig(
      apiKey = sys.env.getOrElse("ANTHROPIC_API_KEY", ""),
      model = sys.env.getOrElse("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
      temperature = sys.env.getOrElse("ANTHROPIC_TEMPERATURE", "0.7").toDouble,
      maxTokens = sys.env.get("ANTHROPIC_MAX_TOKENS").map(_.toInt),
      timeout = Duration.fromMillis(
        sys.env.getOrElse("ANTHROPIC_TIMEOUT_MS", "60000").toLong
      ),
      enableStreaming = sys.env.getOrElse("ANTHROPIC_ENABLE_STREAMING", "true").toBoolean,
      logRequests = sys.env.getOrElse("ANTHROPIC_LOG_REQUESTS", "false").toBoolean,
      logResponses = sys.env.getOrElse("ANTHROPIC_LOG_RESPONSES", "false").toBoolean
    )
  
  /**
   * Creates a ZLayer that provides an AnthropicConfig from environment variables.
   */
  val layer: ULayer[AnthropicConfig] = ZLayer.succeed(createFromEnv)
  val fromEnv: ULayer[AnthropicConfig] = 
    ZLayer.succeed(
      AnthropicConfig(
        apiKey = sys.env.getOrElse("ANTHROPIC_API_KEY", ""),
        model = sys.env.getOrElse("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        temperature = sys.env.getOrElse("ANTHROPIC_TEMPERATURE", "0.7").toDouble,
        maxTokens = sys.env.get("ANTHROPIC_MAX_TOKENS").map(_.toInt),
        timeout = Duration.fromMillis(
          sys.env.getOrElse("ANTHROPIC_TIMEOUT_MS", "60000").toLong
        ),
        enableStreaming = sys.env.getOrElse("ANTHROPIC_ENABLE_STREAMING", "true").toBoolean,
        logRequests = sys.env.getOrElse("ANTHROPIC_LOG_REQUESTS", "false").toBoolean,
        logResponses = sys.env.getOrElse("ANTHROPIC_LOG_RESPONSES", "false").toBoolean
      )
    )