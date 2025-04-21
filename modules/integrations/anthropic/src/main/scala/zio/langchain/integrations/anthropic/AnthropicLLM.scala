package zio.langchain.integrations.anthropic

import zio.*
import zio.json.*
import zio.stream.ZStream
import zio.http.*
import zio.config.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.core.tool.Tool

/**
 * Implementation of the LLM interface for Anthropic Claude models.
 * This version uses ZIO HTTP to connect to Anthropic API.
 *
 * @param config The Anthropic configuration
 */
class AnthropicLLM(config: AnthropicConfig) extends LLM:
  import AnthropicLLM.*
  
  private val apiUrl = "https://api.anthropic.com/v1/messages"
  
  /**
   * Makes an HTTP request to the Anthropic API using ZIO HTTP
   */
  private def makeRequest(jsonBody: String): ZIO[Any, Throwable, String] = {
    // Log request if enabled
    if (config.logRequests) {
      println(s"Anthropic API Request: $jsonBody")
    }
    
    // Create headers
    val headers = Headers(
      Header.ContentType(MediaType.application.json),
      Header.Authorization.Bearer(config.apiKey),
      Header.Custom("anthropic-version", "2023-06-01")
    )
    
    // Create request body
    val body = Body.fromString(jsonBody)
    
    // Create and send request
    ZIO.scoped {
      for {
        // Get client
        client <- ZIO.service[Client].provide(Client.default)
        
        // Parse URL
        url <- ZIO.fromEither(URL.decode(apiUrl))
                 .orElseFail(new RuntimeException(s"Invalid URL: $apiUrl"))
        
        // Create request
        request = Request.post(body, url).updateHeaders(_ ++ headers)
        
        // Send request
        response <- client.request(request)
                      .timeoutFail(new RuntimeException("Request timed out"))(config.timeout)
        
        // Check for errors
        _ <- ZIO.when(response.status.isError) {
          response.body.asString.flatMap { body =>
            ZIO.fail(new RuntimeException(s"Anthropic API error: ${response.status.code}, body: $body"))
          }
        }
        
        // Get response body
        responseBody <- response.body.asString
        
        // Log response if enabled
        _ <- ZIO.when(config.logResponses) {
          ZIO.succeed(println(s"Anthropic API Response: $responseBody"))
        }
      } yield responseBody
    }
  }
  
  /**
   * Completes a text prompt.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a string completion or fails with an LLMError
   */
  override def complete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, String] =
    // Convert to chat completion with a single user message
    completeChat(Seq(ChatMessage.user(prompt)), parameters)
      .map(_.message.contentAsString)
  
  /**
   * Completes a chat conversation.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  override def completeChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse] =
    // Create request
    val apiMessages = messages.map { message =>
      message.role match
        case Role.User => ApiMessage("user", message.contentAsString)
        case Role.Assistant => ApiMessage("assistant", message.contentAsString)
        case Role.System => ApiMessage("system", message.contentAsString)
        case _ => ApiMessage("user", message.contentAsString) // Default to user for unsupported roles
    }
    
    val temperature = parameters
      .flatMap(_.asInstanceOf[DefaultModelParameters].temperature)
      .getOrElse(config.temperature)
    
    val maxTokens = parameters
      .flatMap(_.asInstanceOf[DefaultModelParameters].maxTokens)
      .orElse(config.maxTokens)
    
    // Extract system message if present
    val (systemPrompt, userMessages) = extractSystemMessage(apiMessages)
    
    val request = MessageRequest(
      model = config.model,
      messages = userMessages,
      system = systemPrompt,
      temperature = temperature,
      max_tokens = maxTokens.getOrElse(1024)
    )
    
    val jsonBody = request.toJson
    
    // Make request and parse response
    val result = for
      respBody <- makeRequest(jsonBody)
      respObj <- ZIO.fromEither(respBody.fromJson[MessageResponse])
        .mapError(err => new RuntimeException(s"Failed to parse response: $err"))
      response = createChatResponse(respObj)
    yield response
    
    // Handle errors
    result.mapError(err => LLMError(err))
  
  /**
   * Creates a ChatResponse from the API response.
   */
  private def createChatResponse(response: MessageResponse): ChatResponse =
    val content = response.content.head.text
    
    val message = ChatMessage(
      role = Role.Assistant,
      content = Some(content)
    )
    
    val usage = TokenUsage(
      promptTokens = response.usage.input_tokens,
      completionTokens = response.usage.output_tokens,
      totalTokens = response.usage.input_tokens + response.usage.output_tokens
    )
    
    ChatResponse(
      message = message,
      usage = usage,
      finishReason = Some(response.stop_reason)
    )
  
  /**
   * Extracts the system message from a sequence of API messages.
   * 
   * @param messages The sequence of API messages
   * @return A tuple of (Option[String], Seq[ApiMessage]) containing the system message and the remaining messages
   */
  private def extractSystemMessage(messages: Seq[ApiMessage]): (Option[String], Seq[ApiMessage]) =
    val systemMessages = messages.filter(_.role == "system")
    val otherMessages = messages.filter(_.role != "system")
    
    val systemPrompt = systemMessages.headOption.map(_.content)
    
    (systemPrompt, otherMessages)
  
  /**
   * Completes a chat conversation with function calling capabilities.
   * This is a simplified implementation that doesn't actually use functions.
   */
  override def completeChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse] =
    // For now, just use regular chat completion
    completeChat(messages, parameters)
  
  /**
   * Completes a chat conversation with tool usage capabilities.
   * This is a simplified implementation that doesn't actually use tools.
   */
  override def completeChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse] =
    // For now, just use regular chat completion
    completeChat(messages, parameters)
  
  /**
   * Streams a text completion token by token.
   * This is a simplified implementation that doesn't actually stream.
   */
  override def streamComplete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, String] =
    ZStream.fromZIO(complete(prompt, parameters))
  
  /**
   * Streams a chat completion token by token.
   * This is a simplified implementation that doesn't actually stream.
   */
  override def streamCompleteChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    ZStream.fromZIO(completeChat(messages, parameters))
  
  /**
   * Streams a chat completion with function calling capabilities token by token.
   * This is a simplified implementation that doesn't actually stream or use functions.
   */
  override def streamCompleteChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    ZStream.fromZIO(completeChatWithFunctions(messages, functions, forceFunctionCall, parameters))
  
  /**
   * Streams a chat completion with tool usage capabilities token by token.
   * This is a simplified implementation that doesn't actually stream or use tools.
   */
  override def streamCompleteChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    ZStream.fromZIO(completeChatWithTools(messages, tools, forceToolUse, parameters))

/**
 * Companion object for AnthropicLLM.
 */
object AnthropicLLM:
  // API request and response models
  
  /**
   * Represents a message in the Anthropic API format.
   */
  case class ApiMessage(
    role: String,
    content: String
  )
  
  object ApiMessage:
    given JsonEncoder[ApiMessage] = DeriveJsonEncoder.gen[ApiMessage]
    given JsonDecoder[ApiMessage] = DeriveJsonDecoder.gen[ApiMessage]
  
  /**
   * Represents a message request in the Anthropic API format.
   */
  case class MessageRequest(
    model: String,
    messages: Seq[ApiMessage],
    system: Option[String] = None,
    temperature: Double = 0.7,
    max_tokens: Int = 1024,
    stream: Boolean = false
  )
  
  object MessageRequest:
    given JsonEncoder[MessageRequest] = DeriveJsonEncoder.gen[MessageRequest]
  
  /**
   * Represents content in the Anthropic API response.
   */
  case class ContentBlock(
    `type`: String,
    text: String
  )
  
  object ContentBlock:
    given JsonDecoder[ContentBlock] = DeriveJsonDecoder.gen[ContentBlock]
  
  /**
   * Represents token usage in the Anthropic API format.
   */
  case class ApiUsage(
    input_tokens: Int,
    output_tokens: Int
  )
  
  object ApiUsage:
    given JsonDecoder[ApiUsage] = DeriveJsonDecoder.gen[ApiUsage]
  
  /**
   * Represents a message response in the Anthropic API format.
   */
  case class MessageResponse(
    id: String,
    `type`: String,
    role: String,
    content: List[ContentBlock],
    model: String,
    stop_reason: String,
    usage: ApiUsage
  )
  
  object MessageResponse:
    given JsonDecoder[MessageResponse] = DeriveJsonDecoder.gen[MessageResponse]
  
  /**
   * Creates an AnthropicLLM from an AnthropicConfig.
   *
   * @param config The Anthropic configuration
   * @return A ZIO effect that produces an AnthropicLLM
   */
  def make(config: AnthropicConfig): UIO[AnthropicLLM] =
    ZIO.succeed(new AnthropicLLM(config))
  
  /**
   * Creates a ZLayer that provides an LLM implementation using Anthropic.
   *
   * @return A ZLayer that requires an AnthropicConfig and provides an LLM
   */
  val live: ZLayer[AnthropicConfig, Nothing, LLM] =
    ZLayer {
      for
        config <- ZIO.service[AnthropicConfig]
        llm <- make(config)
      yield llm
    }
  
  /**
   * Creates a ZLayer for the AnthropicLLM using configuration.
   *
   * @return A ZLayer that provides an LLM
   */
  val layer: ZLayer[Any, Config.Error, LLM] =
    AnthropicConfig.layer >>> live