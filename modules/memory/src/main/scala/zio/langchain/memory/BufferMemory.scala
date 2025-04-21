package zio.langchain.memory

import zio.*
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * A simple in-memory implementation of the Memory interface.
 * Stores chat messages in a buffer (vector) and provides basic operations to manage the conversation history.
 */
class BufferMemory private (
  messages: Ref[Vector[ChatMessage]],
  maxMessages: Option[Int] = None
) extends Memory:
  /**
   * Adds a message to the conversation history.
   *
   * @param message The chat message to add
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    messages.updateAndGet { currentMessages =>
      val newMessages = currentMessages :+ message
      maxMessages match
        case Some(max) if newMessages.size > max => newMessages.takeRight(max)
        case _ => newMessages
    }.unit
  
  /**
   * Retrieves all messages from the conversation history.
   *
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
    messages.get
  
  /**
   * Clears the conversation history.
   *
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def clear: ZIO[Any, MemoryError, Unit] =
    messages.set(Vector.empty)

/**
 * Companion object for BufferMemory.
 */
object BufferMemory:
  /**
   * Creates a new BufferMemory instance.
   *
   * @param maxMessages The maximum number of messages to store (optional)
   * @return A ZIO effect that produces a BufferMemory
   */
  def make(maxMessages: Option[Int] = None): UIO[BufferMemory] =
    for
      messagesRef <- Ref.make(Vector.empty[ChatMessage])
    yield new BufferMemory(messagesRef, maxMessages)
  
  /**
   * Creates a new BufferMemory instance with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param maxMessages The maximum number of messages to store (optional)
   * @return A ZIO effect that produces a BufferMemory
   */
  def makeWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    maxMessages: Option[Int] = None
  ): UIO[BufferMemory] =
    for
      messagesRef <- Ref.make(initialMessages.toVector)
    yield new BufferMemory(messagesRef, maxMessages)
  
  /**
   * Creates a ZLayer that provides a Memory implementation using BufferMemory.
   *
   * @param maxMessages The maximum number of messages to store (optional)
   * @return A ZLayer that provides a Memory
   */
  def layer(maxMessages: Option[Int] = None): ULayer[Memory] =
    ZLayer.fromZIO(make(maxMessages))
  
  /**
   * Creates a ZLayer that provides a Memory implementation using BufferMemory with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param maxMessages The maximum number of messages to store (optional)
   * @return A ZLayer that provides a Memory
   */
  def layerWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    maxMessages: Option[Int] = None
  ): ULayer[Memory] =
    ZLayer.fromZIO(makeWithInitialMessages(initialMessages, maxMessages))