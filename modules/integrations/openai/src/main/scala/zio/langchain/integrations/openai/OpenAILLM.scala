package zio.langchain.integrations.openai

import zio.*
import zio.json.*
import zio.stream.ZStream
import zio.http.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.core.tool.Tool

/**
 * Implementation of the LLM interface for OpenAI models.
 * This version uses ZIO HTTP to connect to OpenAI API.
 *
 * @param config The OpenAI configuration
 */
class OpenAILLM(config: OpenAIConfig) extends LLM:
  import OpenAILLM.*
  
  private val apiUrl = "https://api.openai.com/v1/chat/completions"
  
  /**
   * Makes an HTTP request to the OpenAI API using ZIO HTTP
   */
  private def makeRequest(jsonBody: String): ZIO[Any, Throwable, String] = {
    // Log request if enabled
    if (config.logRequests) {
      println(s"OpenAI API Request: $jsonBody")
    }
    
    // Create headers
    val headers = Headers(
      Header.ContentType(MediaType.application.json),
      Header.Authorization.Bearer(config.apiKey)
    ) ++ (config.organizationId match {
      case Some(orgId) => Headers(Header.Custom("OpenAI-Organization", orgId))
      case None => Headers.empty
    })
    
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
        request = Request.post(body, url)
        
        // Send request
        response <- client.request(request)
                      .timeoutFail(new RuntimeException("Request timed out"))(config.timeout)
        
        // Check for errors
        _ <- ZIO.when(response.status.isError) {
          response.body.asString.flatMap { body =>
            ZIO.fail(new RuntimeException(s"OpenAI API error: ${response.status.code}, body: $body"))
          }
        }
        
        // Get response body
        responseBody <- response.body.asString
        
        // Log response if enabled
        _ <- ZIO.when(config.logResponses) {
          ZIO.succeed(println(s"OpenAI API Response: $responseBody"))
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
    
    val request = ChatCompletionRequest(
      model = config.model,
      messages = apiMessages,
      temperature = temperature,
      max_tokens = maxTokens
    )
    
    val jsonBody = request.toJson
    
    // Make request and parse response
    val result = for
      respBody <- makeRequest(jsonBody)
      respObj <- ZIO.fromEither(respBody.fromJson[ChatCompletionResponse])
        .mapError(err => new RuntimeException(s"Failed to parse response: $err"))
      response = createChatResponse(respObj)
    yield response
    
    // Handle errors
    result.mapError(err => LLMError(err))
  
  /**
   * Creates a ChatResponse from the API response.
   */
  private def createChatResponse(response: ChatCompletionResponse): ChatResponse =
    val choice = response.choices.head
    val content = choice.message.content
    
    val message = ChatMessage(
      role = Role.Assistant,
      content = Some(content)
    )
    
    val usage = TokenUsage(
      promptTokens = response.usage.prompt_tokens,
      completionTokens = response.usage.completion_tokens,
      totalTokens = response.usage.total_tokens
    )
    
    ChatResponse(
      message = message,
      usage = usage,
      finishReason = Some(choice.finish_reason)
    )
  
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
 * Companion object for OpenAILLM.
 */
object OpenAILLM:
  // API request and response models
  
  /**
   * Represents a message in the OpenAI API format.
   */
  case class ApiMessage(
    role: String,
    content: String
  )
  
  object ApiMessage:
    given JsonEncoder[ApiMessage] = DeriveJsonEncoder.gen[ApiMessage]
    given JsonDecoder[ApiMessage] = DeriveJsonDecoder.gen[ApiMessage]
  
  /**
   * Represents a chat completion request in the OpenAI API format.
   */
  case class ChatCompletionRequest(
    model: String,
    messages: Seq[ApiMessage],
    temperature: Double = 0.7,
    max_tokens: Option[Int] = None,
    stream: Boolean = false
  )
  
  object ChatCompletionRequest:
    given JsonEncoder[ChatCompletionRequest] = DeriveJsonEncoder.gen[ChatCompletionRequest]
  
  /**
   * Represents a chat completion choice in the OpenAI API format.
   */
  case class ChatCompletionChoice(
    index: Int,
    message: ApiMessage,
    finish_reason: String
  )
  
  object ChatCompletionChoice:
    given JsonDecoder[ChatCompletionChoice] = DeriveJsonDecoder.gen[ChatCompletionChoice]
  
  /**
   * Represents token usage in the OpenAI API format.
   */
  case class ApiUsage(
    prompt_tokens: Int,
    completion_tokens: Int,
    total_tokens: Int
  )
  
  object ApiUsage:
    given JsonDecoder[ApiUsage] = DeriveJsonDecoder.gen[ApiUsage]
  
  /**
   * Represents a chat completion response in the OpenAI API format.
   */
  case class ChatCompletionResponse(
    id: String,
    `object`: String,
    created: Long,
    model: String,
    choices: List[ChatCompletionChoice],
    usage: ApiUsage
  )
  
  object ChatCompletionResponse:
    given JsonDecoder[ChatCompletionResponse] = DeriveJsonDecoder.gen[ChatCompletionResponse]
  
  /**
   * Creates an OpenAILLM from an OpenAIConfig.
   *
   * @param config The OpenAI configuration
   * @return A ZIO effect that produces an OpenAILLM
   */
  def make(config: OpenAIConfig): UIO[OpenAILLM] =
    ZIO.succeed(new OpenAILLM(config))
  
  /**
   * Creates a ZLayer that provides an LLM implementation using OpenAI.
   *
   * @return A ZLayer that requires an OpenAIConfig and provides an LLM
   */
  val live: ZLayer[OpenAIConfig, Nothing, LLM] =
    ZLayer {
      for
        config <- ZIO.service[OpenAIConfig]
        llm <- make(config)
      yield llm
    }
  
  /**
   * Creates a ZLayer for the OpenAILLM using environment variables.
   *
   * @return A ZLayer that provides an LLM
   */
  val fromEnv: ZLayer[Any, Nothing, LLM] =
    OpenAIConfig.layer >>> live