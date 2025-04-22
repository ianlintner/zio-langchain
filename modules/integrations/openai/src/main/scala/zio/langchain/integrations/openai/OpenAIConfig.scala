package zio.langchain.integrations.openai

import zio.*

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
  timeout: Duration = Duration.fromSeconds(60),
  enableStreaming: Boolean = true,
  logRequests: Boolean = false,
  logResponses: Boolean = false
) {
  // Add validation method
  def validate: Either[String, OpenAIConfig] =
    if (apiKey.trim.isEmpty) Left("OpenAI API key is missing or empty")
    else Right(this)
}

/**
 * Companion object for OpenAIConfig.
 */
object OpenAIConfig:
  /**
   * Creates an OpenAIConfig from environment variables.
   */
  /**
   * Creates an OpenAIConfig from environment variables with validation.
   */
  def fromEnv: ZIO[Any, String, OpenAIConfig] =
    ZIO.attempt {
      OpenAIConfig(
        apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
        model = sys.env.getOrElse("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature = sys.env.getOrElse("OPENAI_TEMPERATURE", "0.7").toDouble,
        maxTokens = sys.env.get("OPENAI_MAX_TOKENS").map(_.toInt),
        organizationId = sys.env.get("OPENAI_ORG_ID"),
        timeout = Duration.fromMillis(
          sys.env.getOrElse("OPENAI_TIMEOUT_MS", "60000").toLong
        ),
        enableStreaming = sys.env.getOrElse("OPENAI_ENABLE_STREAMING", "true").toBoolean,
        logRequests = sys.env.getOrElse("OPENAI_LOG_REQUESTS", "false").toBoolean,
        logResponses = sys.env.getOrElse("OPENAI_LOG_RESPONSES", "false").toBoolean
      )
    }.flatMap(config => ZIO.fromEither(config.validate))
  
  /**
   * Creates a ZLayer that provides an OpenAIConfig from environment variables.
   */
  /**
   * Creates a ZLayer that provides a validated OpenAIConfig from environment variables.
   */
  val layer: ZLayer[Any, String, OpenAIConfig] = ZLayer.fromZIO(fromEnv)