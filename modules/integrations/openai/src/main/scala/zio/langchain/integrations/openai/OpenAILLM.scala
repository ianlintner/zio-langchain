package zio.langchain.integrations.openai

import zio.*
import zio.stream.ZStream

import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.output.Response
import dev.langchain4j.data.message.{AiMessage, ChatMessage => LC4JChatMessage, UserMessage, SystemMessage}

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

import scala.jdk.CollectionConverters.*
import java.util.concurrent.TimeUnit

/**
 * Implementation of the LLM interface for OpenAI models.
 *
 * @param client The langchain4j OpenAI client
 * @param config The OpenAI configuration
 */
class OpenAILLM(
  client: ChatLanguageModel,
  config: OpenAIConfig
) extends LLM:
  /**
   * Completes a text prompt.
   *
   * @param prompt The text prompt to complete
   * @return A ZIO effect that produces a string completion or fails with an LLMError
   */
  override def complete(prompt: String): ZIO[Any, LLMError, String] =
    ZIO.attemptBlockingIO {
      client.generate(prompt)
    }.mapError(e => LLMError(e))
  
  /**
   * Completes a chat conversation.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @return A ZIO effect that produces a ChatResponse or fails with an LLMError
   */
  override def completeChat(messages: Seq[domain.ChatMessage]): ZIO[Any, LLMError, domain.ChatResponse] =
    ZIO.attemptBlockingIO {
      val javaMessages = messages.map(convertToJavaMessage).asJava
      val response = client.generate(javaMessages)
      convertFromJavaResponse(response)
    }.mapError(e => LLMError(e))
  
  /**
   * Streams a text completion token by token.
   *
   * @param prompt The text prompt to complete
   * @return A ZStream that produces string tokens or fails with an LLMError
   */
  override def streamComplete(prompt: String): ZStream[Any, LLMError, String] =
    ZStream.fromZIO(complete(prompt))
  
  /**
   * Streams a chat completion token by token.
   *
   * @param messages The sequence of chat messages representing the conversation history
   * @return A ZStream that produces ChatResponse chunks or fails with an LLMError
   */
  override def streamCompleteChat(messages: Seq[domain.ChatMessage]): ZStream[Any, LLMError, domain.ChatResponse] =
    ZStream.fromZIO(completeChat(messages))
  
  /**
   * Converts a ZIO LangChain ChatMessage to a langchain4j ChatMessage.
   *
   * @param message The ZIO LangChain ChatMessage to convert
   * @return The langchain4j ChatMessage
   */
  private def convertToJavaMessage(message: domain.ChatMessage): LC4JChatMessage =
    message.role match
      case domain.Role.User =>
        UserMessage.userMessage(message.content)
      case domain.Role.Assistant =>
        AiMessage.aiMessage(message.content)
      case domain.Role.System =>
        SystemMessage.systemMessage(message.content)
      case domain.Role.Tool =>
        // langchain4j doesn't have a direct equivalent for tool messages
        // so we'll use a user message with a prefix
        UserMessage.userMessage(s"[TOOL] ${message.content}")
  
  /**
   * Converts a langchain4j AiMessage to a ZIO LangChain ChatResponse.
   *
   * @param response The langchain4j Response<AiMessage> to convert
   * @return The ZIO LangChain ChatResponse
   */
  private def convertFromJavaResponse(response: Response[AiMessage]): domain.ChatResponse =
    val message = domain.ChatMessage(
      role = domain.Role.Assistant,
      content = response.content().text(),
      metadata = Map.empty
    )
    
    val tokenUsage = domain.TokenUsage(
      promptTokens = response.tokenUsage().map(_.inputTokenCount()).orElse(0),
      completionTokens = response.tokenUsage().map(_.outputTokenCount()).orElse(0),
      totalTokens = response.tokenUsage().map(_.totalTokenCount()).orElse(0)
    )
    
    domain.ChatResponse(
      message = message,
      usage = tokenUsage,
      finishReason = response.finishReason().map(_.name())
    )

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
      val client = OpenAiChatModel.builder()
        .apiKey(config.apiKey)
        .modelName(config.model)
        .temperature(config.temperature)
        .maxTokens(config.maxTokens.map(Integer.valueOf).orNull)
        .organizationId(config.organizationId.orNull)
        .timeout(config.timeout.toMillis, TimeUnit.MILLISECONDS)
        .build()
      
      new OpenAILLM(client, config)
    }
  
  /**
   * Creates a ZLayer that provides an LLM implementation using OpenAI.
   *
   * @return A ZLayer that requires an OpenAIConfig and provides an LLM
   */
  val live: ZLayer[OpenAIConfig, Throwable, LLM] =
    ZLayer {
      for
        config <- ZIO.service[OpenAIConfig]
        llm <- make(config)
      yield llm
    }