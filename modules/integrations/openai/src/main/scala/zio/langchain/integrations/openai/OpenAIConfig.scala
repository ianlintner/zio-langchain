package zio.langchain.integrations.openai

import zio.*

/**
 * Configuration for the OpenAI API.
 *
 * @param apiKey The API key for the OpenAI API
 * @param model The model to use for completions
 * @param temperature The temperature to use for completions
 * @param maxTokens The maximum number of tokens to generate
 * @param organizationId The organization ID to use for the OpenAI API
 * @param timeout The timeout for API requests
 * @param enableStreaming Whether to enable streaming for completions
 * @param logRequests Whether to log requests to the OpenAI API
 * @param logResponses Whether to log responses from the OpenAI API
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

object OpenAIConfig:
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
    }.catchAll(ex => ZIO.fail(s"Error creating OpenAIConfig: ${ex.getMessage}"))
     .flatMap(config => ZIO.fromEither(config.validate))
  
  /**
   * Creates a ZLayer that provides a validated OpenAIConfig from environment variables.
   */
  val layer: ZLayer[Any, String, OpenAIConfig] = ZLayer.fromZIO(fromEnv)