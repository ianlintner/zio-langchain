package zio.langchain.memory

import zio.*
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.core.model.LLM

/**
 * A memory implementation that uses an LLM to periodically summarize conversation history.
 * This helps maintain context for long conversations while keeping memory usage manageable.
 *
 * @param messages The reference to the current messages
 * @param summary The reference to the current summary
 * @param llm The LLM used for summarization
 * @param maxMessages The maximum number of messages to keep before summarizing
 * @param summarizationThreshold The number of messages that triggers a summarization
 * @param summarizationPrompt The prompt template used for summarization
 */
class SummaryMemory private (
  messages: Ref[Vector[ChatMessage]],
  summary: Ref[Option[String]],
  llm: LLM,
  maxMessages: Option[Int] = None,
  summarizationThreshold: Int = 10,
  summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
) extends Memory:
  /**
   * Adds a message to the conversation history and triggers summarization if needed.
   *
   * @param message The chat message to add
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    for {
      // Add message to the buffer
      updatedMessages <- messages.updateAndGet { currentMessages =>
        val newMessages = currentMessages :+ message
        maxMessages match
          case Some(max) if newMessages.size > max => newMessages.takeRight(max)
          case _ => newMessages
      }
      
      // Check if we need to summarize
      _ <- ZIO.when(updatedMessages.size >= summarizationThreshold) {
        summarizeMessages(updatedMessages)
      }
    } yield ()
  
  /**
   * Retrieves all messages from the conversation history, including the summary if available.
   *
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
    for {
      currentMessages <- messages.get
      currentSummary <- summary.get
      result = currentSummary match {
        case Some(summaryText) => 
          // Add the summary as a system message at the beginning
          ChatMessage.system(s"Previous conversation summary: $summaryText") +: currentMessages
        case None => 
          currentMessages
      }
    } yield result
  
  /**
   * Clears the conversation history and summary.
   *
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def clear: ZIO[Any, MemoryError, Unit] =
    for {
      _ <- messages.set(Vector.empty)
      _ <- summary.set(None)
    } yield ()
  
  /**
   * Summarizes the current messages and updates the summary.
   *
   * @param messagesToSummarize The messages to summarize
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  private def summarizeMessages(messagesToSummarize: Vector[ChatMessage]): ZIO[Any, MemoryError, Unit] =
    for {
      // Format messages for the summarization prompt
      formattedMessages <- ZIO.succeed(messagesToSummarize.map { msg =>
        s"${msg.role}: ${msg.contentAsString}"
      }.mkString("\n"))
      
      // Replace placeholder in the prompt template
      prompt = summarizationPrompt.replace("{messages}", formattedMessages)
      
      // Generate summary using the LLM
      summaryText <- llm.complete(prompt)
        .mapError(err => MemoryError(err, "Failed to generate conversation summary"))
      
      // Update the summary
      _ <- summary.set(Some(summaryText))
      
      // Clear the message buffer, keeping only the most recent message
      _ <- messages.update(_.takeRight(1))
    } yield ()

/**
 * Companion object for SummaryMemory.
 */
object SummaryMemory:
  /**
   * Creates a new SummaryMemory instance.
   *
   * @param llm The LLM to use for summarization
   * @param maxMessages The maximum number of messages to store (optional)
   * @param summarizationThreshold The number of messages that triggers a summarization
   * @param summarizationPrompt The prompt template used for summarization
   * @return A ZIO effect that produces a SummaryMemory
   */
  def make(
    llm: LLM,
    maxMessages: Option[Int] = None,
    summarizationThreshold: Int = 10,
    summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
  ): UIO[SummaryMemory] =
    for {
      messagesRef <- Ref.make(Vector.empty[ChatMessage])
      summaryRef <- Ref.make(Option.empty[String])
    } yield new SummaryMemory(
      messagesRef, 
      summaryRef, 
      llm, 
      maxMessages, 
      summarizationThreshold, 
      summarizationPrompt
    )
  
  /**
   * Creates a new SummaryMemory instance with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param llm The LLM to use for summarization
   * @param maxMessages The maximum number of messages to store (optional)
   * @param summarizationThreshold The number of messages that triggers a summarization
   * @param summarizationPrompt The prompt template used for summarization
   * @return A ZIO effect that produces a SummaryMemory
   */
  def makeWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    llm: LLM,
    maxMessages: Option[Int] = None,
    summarizationThreshold: Int = 10,
    summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
  ): UIO[SummaryMemory] =
    for {
      messagesRef <- Ref.make(initialMessages.toVector)
      summaryRef <- Ref.make(Option.empty[String])
    } yield new SummaryMemory(
      messagesRef, 
      summaryRef, 
      llm, 
      maxMessages, 
      summarizationThreshold, 
      summarizationPrompt
    )
  
  /**
   * Creates a ZLayer that provides a Memory implementation using SummaryMemory.
   *
   * @param llm The LLM to use for summarization
   * @param maxMessages The maximum number of messages to store (optional)
   * @param summarizationThreshold The number of messages that triggers a summarization
   * @param summarizationPrompt The prompt template used for summarization
   * @return A ZLayer that provides a Memory
   */
  def layer(
    llm: LLM,
    maxMessages: Option[Int] = None,
    summarizationThreshold: Int = 10,
    summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
  ): ULayer[Memory] =
    ZLayer.fromZIO(make(llm, maxMessages, summarizationThreshold, summarizationPrompt))
  
  /**
   * Creates a ZLayer that provides a Memory implementation using SummaryMemory with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param llm The LLM to use for summarization
   * @param maxMessages The maximum number of messages to store (optional)
   * @param summarizationThreshold The number of messages that triggers a summarization
   * @param summarizationPrompt The prompt template used for summarization
   * @return A ZLayer that provides a Memory
   */
  def layerWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    llm: LLM,
    maxMessages: Option[Int] = None,
    summarizationThreshold: Int = 10,
    summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
  ): ULayer[Memory] =
    ZLayer.fromZIO(makeWithInitialMessages(initialMessages, llm, maxMessages, summarizationThreshold, summarizationPrompt))
  
  /**
   * Creates a ZLayer that provides a Memory implementation using SummaryMemory.
   * This version takes an LLM from the environment.
   *
   * @param maxMessages The maximum number of messages to store (optional)
   * @param summarizationThreshold The number of messages that triggers a summarization
   * @param summarizationPrompt The prompt template used for summarization
   * @return A ZLayer that requires an LLM and provides a Memory
   */
  def layerFromLLM(
    maxMessages: Option[Int] = None,
    summarizationThreshold: Int = 10,
    summarizationPrompt: String = "Summarize the following conversation concisely while preserving all important information:\n\n{messages}"
  ): ZLayer[LLM, Nothing, Memory] =
    ZLayer.fromZIO(
      for {
        llm <- ZIO.service[LLM]
        memory <- make(llm, maxMessages, summarizationThreshold, summarizationPrompt)
      } yield memory
    )