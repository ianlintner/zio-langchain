package zio.langchain.integrations.openai

import zio.*
import zio.stream.ZStream

import dev.langchain4j.model.openai.{OpenAiChatModel, OpenAiStreamingChatModel}
import dev.langchain4j.model.chat.{ChatLanguageModel, StreamingChatLanguageModel}
import dev.langchain4j.model.output.Response
import dev.langchain4j.data.message.{AiMessage, ChatMessage => LC4JChatMessage, UserMessage, SystemMessage}
import zio.langchain.core.domain
import dev.langchain4j.model.openai.{OpenAiModelName, OpenAiFunctionCallMode}
import dev.langchain4j.agent.tool.{ToolSpecification, ToolExecutionRequest}

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

import scala.jdk.CollectionConverters.*
import scala.jdk.OptionConverters.*
import java.util.concurrent.TimeUnit
import java.util.{List => JList, Map => JMap}
import scala.collection.mutable.ArrayBuffer
import java.time.Duration
import scala.compiletime.asMatchable

/**
 * Implementation of the LLM interface for OpenAI models.
 *
 * @param chatClient The langchain4j OpenAI chat client
 * @param streamingChatClient The langchain4j OpenAI streaming chat client
 * @param config The OpenAI configuration
 */
class OpenAILLM(
  chatClient: ChatLanguageModel,
  streamingChatClient: Option[StreamingChatLanguageModel],
  config: OpenAIConfig
) extends LLM:
  
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
    ZIO.attemptBlockingIO {
      chatClient.generate(prompt)
    }.retry(retrySchedule)
     .mapError(e => LLMError(e))
  
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
    ZIO.attemptBlockingIO {
      val javaMessages = messages.map(convertToJavaMessage).asJava
      val response = parameters match
        case Some(params) =>
          val modelOptions = createModelOptions(params)
          chatClient.generate(javaMessages, modelOptions)
        case None =>
          chatClient.generate(javaMessages)
          
      convertFromJavaResponse(response)
    }.retry(retrySchedule)
     .mapError(e => LLMError(e))
  
  /**
   * Completes a chat conversation with function calling capabilities.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If provided, force the model to call the specified function
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  override def completeChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse] =
    ZIO.attemptBlockingIO {
      val javaMessages = messages.map(convertToJavaMessage).asJava
      val javaFunctions = functions.map(convertToJavaToolSpecification).asJava
      
      val functionCallMode = forceFunctionCall match
        case Some(functionName) => OpenAiFunctionCallMode.FORCING.withFunctionName(functionName)
        case None => OpenAiFunctionCallMode.AUTO
      
      val response = parameters match
        case Some(params) =>
          val modelOptions = createModelOptions(params)
          chatClient.generate(javaMessages, javaFunctions, functionCallMode, modelOptions)
        case None =>
          chatClient.generate(javaMessages, javaFunctions, functionCallMode)
          
      convertFromJavaResponse(response)
    }.retry(retrySchedule)
     .mapError(e => LLMError(e))
  
  /**
   * Completes a chat conversation with tool usage capabilities.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  override def completeChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse] =
    completeChatWithFunctions(
      messages,
      tools.map(tool => tool.function),
      if (forceToolUse && tools.nonEmpty) Some(tools.head.function.name) else None,
      parameters
    )
  
  /**
   * Streams a text completion token by token.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces string tokens or fails with an LLMError
   */
  override def streamComplete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, String] =
    streamingChatClient match
      case Some(client) =>
        ZStream.async[Any, LLMError, String] { emit =>
          ZIO.attemptBlockingIO {
            val handler = new dev.langchain4j.model.output.TokenStreamHandler {
              override def onNext(token: String): Unit =
                emit(ZIO.succeed(Chunk.single(token)))
              
              override def onComplete(): Unit =
                emit(ZIO.succeed(Chunk.empty))
              
              override def onError(error: Throwable): Unit =
                emit(ZIO.fail(LLMError(error)))
            }
            
            parameters match
              case Some(params) =>
                val modelOptions = createModelOptions(params)
                client.generate(prompt, handler, modelOptions)
              case None =>
                client.generate(prompt, handler)
          }.catchAll { error =>
            ZIO.succeed(emit(ZIO.fail(LLMError(error))))
          }
        }.retry(retrySchedule)
      
      case None =>
        // Fall back to non-streaming if streaming client is not available
        ZStream.fromZIO(complete(prompt, parameters))
  
  /**
   * Streams a chat completion token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  override def streamCompleteChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    streamingChatClient match
      case Some(client) =>
        ZStream.async[Any, LLMError, domain.ChatResponse] { emit =>
          ZIO.attemptBlockingIO {
            val javaMessages = messages.map(convertToJavaMessage).asJava
            val tokenBuffer = new StringBuilder()
            
            val handler = new dev.langchain4j.model.output.TokenStreamHandler {
              override def onNext(token: String): Unit =
                tokenBuffer.append(token)
                val partialMessage = ChatMessage(
                  role = Role.Assistant,
                  content = Some(tokenBuffer.toString),
                  metadata = Map.empty
                )
                
                val partialResponse = ChatResponse(
                  message = partialMessage,
                  usage = TokenUsage(0, 0, 0),
                  finishReason = None
                )
                
                emit(ZIO.succeed(Chunk.single(partialResponse)))
              
              override def onComplete(): Unit =
                emit(ZIO.succeed(Chunk.empty))
              
              override def onError(error: Throwable): Unit =
                emit(ZIO.fail(LLMError(error)))
            }
            
            parameters match
              case Some(params) =>
                val modelOptions = createModelOptions(params)
                client.generate(javaMessages, handler, modelOptions)
              case None =>
                client.generate(javaMessages, handler)
          }.catchAll { error =>
            ZIO.succeed(emit(ZIO.fail(LLMError(error))))
          }
        }.retry(retrySchedule)
        
      case None =>
        // Fall back to non-streaming if streaming client is not available
        ZStream.fromZIO(completeChat(messages, parameters))
  
  /**
   * Streams a chat completion with function calling capabilities token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If provided, force the model to call the specified function
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  override def streamCompleteChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    // Current version of langchain4j doesn't fully support streaming with function calling
    // This is a fallback implementation that uses the non-streaming API
    ZStream.fromZIO(completeChatWithFunctions(messages, functions, forceFunctionCall, parameters))
  
  /**
   * Streams a chat completion with tool usage capabilities token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  override def streamCompleteChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse] =
    streamCompleteChatWithFunctions(
      messages,
      tools.map(_.function),
      if (forceToolUse && tools.nonEmpty) Some(tools.head.function.name) else None,
      parameters
    )
  
  /**
   * Creates model options from ModelParameters.
   *
   * @param parameters The model parameters to convert
   * @return A map of model options
   */
  private def createModelOptions(parameters: ModelParameters): JMap[String, Object] =
    val javaMap = new java.util.HashMap[String, Object]()
    
    parameters.toMap.foreach { (key, value) =>
      val javaValue = value.asMatchable match
        case v: Double => java.lang.Double.valueOf(v)
        case v: Int => java.lang.Integer.valueOf(v)
        case v: Boolean => java.lang.Boolean.valueOf(v)
        case v: String => v
        case v: Long => java.lang.Long.valueOf(v)
        case _ => value.toString
      
      javaMap.put(key, javaValue)
    }
    
    javaMap
  
  /**
   * Converts a FunctionDefinition to a ToolSpecification for langchain4j.
   *
   * @param function The function definition to convert
   * @return The langchain4j ToolSpecification
   */
  private def convertToJavaToolSpecification(function: FunctionDefinition): ToolSpecification =
    val parametersMap = new java.util.HashMap[String, Object]()
    val properties = new java.util.HashMap[String, Object]()
    val required = new java.util.ArrayList[String]()
    
    function.parameters.foreach { param =>
      val paramProperties = new java.util.HashMap[String, Object]()
      paramProperties.put("type", param.`type`)
      
      if param.description.nonEmpty then
        paramProperties.put("description", param.description)
      
      if param.possibleValues.nonEmpty then
        val enumValues = param.possibleValues.get.asJava
        paramProperties.put("enum", enumValues)
      
      properties.put(param.name, paramProperties)
      
      if param.required then
        required.add(param.name)
    }
    
    parametersMap.put("type", "object")
    parametersMap.put("properties", properties)
    
    if required.size() > 0 then
      parametersMap.put("required", required)
    
    // Create a ToolParameters instance from our map
    val toolParams = new dev.langchain4j.agent.tool.ToolParameters(parametersMap)
    
    ToolSpecification.builder()
      .name(function.name)
      .description(function.description)
      .parameters(toolParams)
      .build()
  
  /**
   * Converts a ZIO LangChain ChatMessage to a langchain4j ChatMessage.
   *
   * @param message The ZIO LangChain ChatMessage to convert
   * @return The langchain4j ChatMessage
   */
  private def convertToJavaMessage(message: domain.ChatMessage): LC4JChatMessage =
    // Simplified conversion that works with basic roles
    message.role match
      case Role.User =>
        UserMessage.userMessage(message.contentAsString)
      
      case Role.Assistant =>
        AiMessage.aiMessage(message.contentAsString)
      
      case Role.System =>
        SystemMessage.systemMessage(message.contentAsString)
      
      case _ =>
        // Default to user message for other roles as fallback
        UserMessage.userMessage(message.contentAsString)
  
  /**
   * Converts a langchain4j AiMessage to a ZIO LangChain ChatResponse.
   *
   * @param response The langchain4j Response<AiMessage> to convert
   * @return The ZIO LangChain ChatResponse
   */
  private def convertFromJavaResponse(response: Response[AiMessage]): domain.ChatResponse =
    val aiMessage = response.content()
    
    // Simplified conversion without function call checks
    val message = ChatMessage(
      role = Role.Assistant,
      content = Some(aiMessage.text()),
      metadata = Map.empty
    )
    
    // Safely extract token usage information
    val tokenUsage = {
      if (response.tokenUsage().isPresent) {
        val usage = response.tokenUsage().get()
        TokenUsage(
          promptTokens = usage.inputTokenCount(),
          completionTokens = usage.outputTokenCount(),
          totalTokens = usage.totalTokenCount()
        )
      } else {
        TokenUsage(0, 0, 0)
      }
    }
    
    // Safely extract finish reason
    val finishReason = if (response.finishReason().isPresent) {
      Some(response.finishReason().get().name())
    } else {
      None
    }
    
    ChatResponse(
      message = message,
      usage = tokenUsage,
      finishReason = finishReason
    )
  
  /**
   * The retry schedule for API calls.
   */
  private val retrySchedule = Schedule.exponential(100.milliseconds) && Schedule.recurs(3)

/**
 * Companion object for OpenAILLM.
 */
object OpenAILLM:
  /**
   * Creates an OpenAILLM from an OpenAIConfig.
   *
   * @param config The OpenAI configuration
   * @return A ZIO effect that produces an OpenAILLM or fails with a Throwable
   */
  def make(config: OpenAIConfig): ZIO[Any, Throwable, OpenAILLM] =
    ZIO.attempt {
      val chatClient = OpenAiChatModel.builder()
        .apiKey(config.apiKey)
        .modelName(config.model)
        .temperature(config.temperature)
        .maxTokens(config.maxTokens.map(Integer.valueOf).orNull)
        // .organizationId(config.organizationId.orNull) // Not supported in this version
        .timeout(Duration.ofMillis(config.timeout.toMillis))
        .logRequests(config.logRequests)
        .logResponses(config.logResponses)
        .build()
      
      val streamingChatClient =
        if config.enableStreaming then
          Some(OpenAiStreamingChatModel.builder()
            .apiKey(config.apiKey)
            .modelName(config.model)
            .temperature(config.temperature)
            .maxTokens(config.maxTokens.map(Integer.valueOf).orNull)
            // .organizationId(config.organizationId.orNull) // Not supported in this version
            .timeout(Duration.ofMillis(config.timeout.toMillis))
            .logRequests(config.logRequests)
            .logResponses(config.logResponses)
            .build())
        else None
      
      new OpenAILLM(chatClient, streamingChatClient, config)
    }
  
  /**
   * Creates a ZLayer that provides an LLM implementation using OpenAI.
   *
   * @return A ZLayer that requires an OpenAIConfig and provides an LLM
   */
  val live: ZLayer[OpenAIConfig, Throwable, LLM] =
    ZLayer {
      for
        config <- ZIO.serviceWith[OpenAIConfig](identity)
        llm <- make(config)
      yield llm
    }