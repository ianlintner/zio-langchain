package zio.langchain.core.model

import zio.*
import zio.stream.ZStream

import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.core.tool.Tool

/**
 * Interface for Large Language Models (LLMs).
 * Provides methods for text completion and chat completion with both synchronous and streaming variants.
 */
trait LLM:
  /**
   * Completes a text prompt.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a string completion or fails with an LLMError
   */
  def complete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, String]
  
  /**
   * Completes a chat conversation.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  def completeChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse]
  
  /**
   * Completes a chat conversation with function calling capabilities.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If true, force the model to call a function
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  def completeChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse]
  
  /**
   * Completes a chat conversation with tool usage capabilities.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  def completeChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZIO[Any, LLMError, ChatResponse]
  
  /**
   * Streams a text completion token by token.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces string tokens or fails with an LLMError
   */
  def streamComplete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, String]
  
  /**
   * Streams a chat completion token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse]
  
  /**
   * Streams a chat completion with function calling capabilities token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If provided, force the model to call the specified function
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse]
  
  /**
   * Streams a chat completion with tool usage capabilities token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZStream[Any, LLMError, ChatResponse]

/**
 * Companion object for LLM.
 */
object LLM:
  /**
   * Creates a ZIO accessor for the LLM service.
   *
   * @return A ZIO effect that requires an LLM and produces the LLM
   */
  def get: ZIO[LLM, Nothing, LLM] = ZIO.service[LLM]
  
  /**
   * Completes a text prompt using the LLM service.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that requires an LLM and produces a string completion or fails with an LLMError
   */
  def complete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZIO[LLM, LLMError, String] =
    ZIO.serviceWithZIO[LLM](_.complete(prompt, parameters))
  
  /**
   * Completes a chat conversation using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that requires an LLM and produces a ChatResponse or fails with an LLMError
   */
  def completeChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZIO[LLM, LLMError, ChatResponse] =
    ZIO.serviceWithZIO[LLM](_.completeChat(messages, parameters))
  
  /**
   * Completes a chat conversation with function calling capabilities using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If provided, force the model to call the specified function
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that requires an LLM and produces a ChatResponse or fails with an LLMError
   */
  def completeChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZIO[LLM, LLMError, ChatResponse] =
    ZIO.serviceWithZIO[LLM](_.completeChatWithFunctions(messages, functions, forceFunctionCall, parameters))
  
  /**
   * Completes a chat conversation with tool usage capabilities using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZIO effect that requires an LLM and produces a ChatResponse or fails with an LLMError
   */
  def completeChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZIO[LLM, LLMError, ChatResponse] =
    ZIO.serviceWithZIO[LLM](_.completeChatWithTools(messages, tools, forceToolUse, parameters))
  
  /**
   * Convert ZIO Tool instances to ToolDefinitions for use with the LLM.
   *
   * @param tools The sequence of Tool instances to convert
   * @return A sequence of ToolDefinition instances
   */
  def convertToolsToDefinitions(tools: Seq[Tool[Any, LangChainError]]): Seq[ToolDefinition] =
    tools.map { tool =>
      ToolDefinition(
        `type` = "function",
        function = FunctionDefinition(
          name = tool.name,
          description = tool.description,
          parameters = Seq(
            FunctionParameter(
              name = "input",
              description = s"Input for the ${tool.name} tool",
              required = true,
              `type` = "string"
            )
          )
        )
      )
    }
  
  /**
   * Streams a text completion token by token using the LLM service.
   *
   * @param prompt The text prompt to complete
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that requires an LLM and produces string tokens or fails with an LLMError
   */
  def streamComplete(
    prompt: String,
    parameters: Option[ModelParameters] = None
  ): ZStream[LLM, LLMError, String] =
    ZStream.serviceWithStream[LLM](_.streamComplete(prompt, parameters))
  
  /**
   * Streams a chat completion token by token using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that requires an LLM and produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChat(
    messages: Seq[ChatMessage],
    parameters: Option[ModelParameters] = None
  ): ZStream[LLM, LLMError, ChatResponse] =
    ZStream.serviceWithStream[LLM](_.streamCompleteChat(messages, parameters))
  
  /**
   * Streams a chat completion with function calling capabilities token by token using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param functions The functions that can be called by the model
   * @param forceFunctionCall If provided, force the model to call the specified function
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that requires an LLM and produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChatWithFunctions(
    messages: Seq[ChatMessage],
    functions: Seq[FunctionDefinition],
    forceFunctionCall: Option[String] = None,
    parameters: Option[ModelParameters] = None
  ): ZStream[LLM, LLMError, ChatResponse] =
    ZStream.serviceWithStream[LLM](_.streamCompleteChatWithFunctions(messages, functions, forceFunctionCall, parameters))
  
  /**
   * Streams a chat completion with tool usage capabilities token by token using the LLM service.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @param tools The tools that can be used by the model
   * @param forceToolUse If true, force the model to use a tool
   * @param parameters Optional model parameters to control the generation
   * @return A ZStream that requires an LLM and produces ChatResponse chunks or fails with an LLMError
   */
  def streamCompleteChatWithTools(
    messages: Seq[ChatMessage],
    tools: Seq[ToolDefinition],
    forceToolUse: Boolean = false,
    parameters: Option[ModelParameters] = None
  ): ZStream[LLM, LLMError, ChatResponse] =
    ZStream.serviceWithStream[LLM](_.streamCompleteChatWithTools(messages, tools, forceToolUse, parameters))