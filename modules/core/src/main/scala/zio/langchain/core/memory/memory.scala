package zio.langchain.core.memory

import zio.*

import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * Interface for conversation memory.
 * Provides methods for storing, retrieving, and managing conversation history.
 */
trait Memory:
  /**
   * Adds a message to the conversation history.
   *
   * @param message The chat message to add
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  def add(message: ChatMessage): ZIO[Any, MemoryError, Unit]
  
  /**
   * Adds multiple messages to the conversation history.
   *
   * @param messages The sequence of chat messages to add
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  def addAll(messages: Seq[ChatMessage]): ZIO[Any, MemoryError, Unit] =
    ZIO.foreachDiscard(messages)(add)
  
  /**
   * Retrieves all messages from the conversation history.
   *
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  def get: ZIO[Any, MemoryError, Seq[ChatMessage]]
  
  /**
   * Clears the conversation history.
   *
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  def clear: ZIO[Any, MemoryError, Unit]
  
  /**
   * Retrieves the most recent messages from the conversation history.
   *
   * @param n The number of most recent messages to retrieve
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  def getRecent(n: Int): ZIO[Any, MemoryError, Seq[ChatMessage]] =
    get.map(_.takeRight(n))

/**
 * Companion object for Memory.
 */
object Memory:
  /**
   * Creates a ZIO accessor for the Memory service.
   *
   * @return A ZIO effect that requires a Memory and produces the Memory
   */
  def get: ZIO[Memory, Nothing, Memory] = ZIO.service[Memory]
  
  /**
   * Adds a message to the conversation history using the Memory service.
   *
   * @param message The chat message to add
   * @return A ZIO effect that requires a Memory and completes with unit or fails with a MemoryError
   */
  def add(message: ChatMessage): ZIO[Memory, MemoryError, Unit] =
    ZIO.serviceWithZIO[Memory](_.add(message))
  
  /**
   * Adds multiple messages to the conversation history using the Memory service.
   *
   * @param messages The sequence of chat messages to add
   * @return A ZIO effect that requires a Memory and completes with unit or fails with a MemoryError
   */
  def addAll(messages: Seq[ChatMessage]): ZIO[Memory, MemoryError, Unit] =
    ZIO.serviceWithZIO[Memory](_.addAll(messages))
  
  /**
   * Retrieves all messages from the conversation history using the Memory service.
   *
   * @return A ZIO effect that requires a Memory and produces a sequence of chat messages or fails with a MemoryError
   */
  def getMessages: ZIO[Memory, MemoryError, Seq[ChatMessage]] =
    ZIO.serviceWithZIO[Memory](_.get)
  
  /**
   * Clears the conversation history using the Memory service.
   *
   * @return A ZIO effect that requires a Memory and completes with unit or fails with a MemoryError
   */
  def clear: ZIO[Memory, MemoryError, Unit] =
    ZIO.serviceWithZIO[Memory](_.clear)
  
  /**
   * Retrieves the most recent messages from the conversation history using the Memory service.
   *
   * @param n The number of most recent messages to retrieve
   * @return A ZIO effect that requires a Memory and produces a sequence of chat messages or fails with a MemoryError
   */
  def getRecent(n: Int): ZIO[Memory, MemoryError, Seq[ChatMessage]] =
    ZIO.serviceWithZIO[Memory](_.getRecent(n))