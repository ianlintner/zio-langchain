---
title: Memory
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Memory

This document explains memory systems in ZIO LangChain - how they store conversation history and maintain state across interactions.

## Table of Contents

- [Introduction](#introduction)
- [Core Memory Interface](#core-memory-interface)
- [Memory Types](#memory-types)
- [Usage Examples](#usage-examples)
- [Building Custom Memory](#building-custom-memory)
- [Best Practices](#best-practices)

## Introduction

Memory components in ZIO LangChain provide mechanisms for:

- Storing conversation history between users and language models
- Maintaining state across multiple interactions
- Providing context for follow-up questions and responses
- Implementing personalized responses based on past interactions
- Managing long-running conversations efficiently

Without memory, each interaction with an LLM would be stateless - the model would have no knowledge of previous exchanges. Memory systems overcome this limitation by storing and retrieving relevant conversation history.

## Core Memory Interface

The foundation of memory functionality in ZIO LangChain is the `Memory` trait:

```scala
trait Memory:
  def add(message: ChatMessage): ZIO[Any, MemoryError, Unit]
  def get: ZIO[Any, MemoryError, Seq[ChatMessage]]
  def clear: ZIO[Any, MemoryError, Unit]
```

This interface is intentionally minimal, providing just the essential operations:

- **add**: Store a new message in memory
- **get**: Retrieve all stored messages
- **clear**: Remove all messages from memory

All operations are represented as ZIO effects, with proper error typing through `MemoryError`.

## Memory Types

ZIO LangChain provides several implementations of the `Memory` trait:

### Volatile Memory

The simplest form of memory, storing conversations in-memory:

```scala
class VolatileMemory extends Memory:
  private val messages = new AtomicReference[Vector[ChatMessage]](Vector.empty)
  
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    ZIO.succeed {
      messages.getAndUpdate(_ :+ message)
      ()
    }
    
  override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
    ZIO.succeed(messages.get())
    
  override def clear: ZIO[Any, MemoryError, Unit] =
    ZIO.succeed {
      messages.set(Vector.empty)
    }
```

Volatile memory is suitable for single-session applications or testing, but data is lost when the application terminates.

### Buffer Memory

Maintains a fixed-size buffer of recent messages, automatically removing older ones:

```scala
class BufferMemory(maxMessages: Int = 10) extends Memory:
  private val messages = new AtomicReference[Vector[ChatMessage]](Vector.empty)
  
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    ZIO.succeed {
      messages.getAndUpdate { msgs =>
        (msgs :+ message).takeRight(maxMessages)
      }
      ()
    }
  
  // Other methods same as VolatileMemory
```

Buffer memory prevents excessive token usage by limiting context size.

### Redis Memory

For persistent, distributed memory across application instances:

```scala
class RedisMemory(
  redis: RedisClient,
  config: RedisConfig
) extends Memory:
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    redis.lpush(config.key, serializeMessage(message))
      .mapError(e => MemoryError(e))
      .unit
      
  override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
    redis.lrange(config.key, 0, -1)
      .mapError(e => MemoryError(e))
      .map(_.map(deserializeMessage))
      
  override def clear: ZIO[Any, MemoryError, Unit] =
    redis.del(config.key)
      .mapError(e => MemoryError(e))
      .unit
      
  private def serializeMessage(message: ChatMessage): String = // Implementation
  private def deserializeMessage(json: String): ChatMessage = // Implementation
```

Redis memory enables stateful conversations across multiple application instances and restarts.

### Summary Memory

Instead of storing all messages, maintains a summary of the conversation:

```scala
class SummaryMemory(llm: LLM) extends Memory:
  private val messages = new AtomicReference[Vector[ChatMessage]](Vector.empty)
  private val summary = new AtomicReference[String]("")
  
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    for
      // Add message to buffer
      _ <- ZIO.succeed {
        messages.getAndUpdate(_ :+ message)
      }
      
      // If buffer reaches threshold, update summary
      currentMessages = messages.get()
      _ <- ZIO.when(currentMessages.size >= 10) {
        updateSummary(currentMessages).mapError(e => MemoryError(e))
      }
    yield ()
    
  private def updateSummary(messages: Seq[ChatMessage]): ZIO[Any, Throwable, Unit] =
    for
      currentSummary <- ZIO.succeed(summary.get())
      
      // Create prompt for summary update
      messagesText = messages.map(m => s"${m.role}: ${m.content}").mkString("\n")
      prompt = s"""
                  |Current conversation summary:
                  |$currentSummary
                  |
                  |New messages:
                  |$messagesText
                  |
                  |Updated summary of the conversation:
                  |""".stripMargin
      
      // Get new summary from LLM
      newSummary <- llm.complete(prompt)
      
      // Update summary and clear message buffer
      _ <- ZIO.succeed {
        summary.set(newSummary)
        messages.set(Vector.empty)
      }
    yield ()
  
  // Other methods with appropriate implementation
```

Summary memory is useful for very long conversations that would exceed token limits.

## Usage Examples

### Simple Chat With Memory

```scala
import zio.*
import zio.langchain.core.model.LLM
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*

object ChatWithMemory extends ZIOAppDefault:
  override def run =
    for
      // Get dependencies
      llm <- ZIO.service[LLM]
      memory <- ZIO.service[Memory]
      
      // Initialize conversation
      _ <- memory.add(ChatMessage(Role.System, "You are a helpful assistant."))
      
      // Chat loop
      _ <- Console.printLine("Chat started. Type 'exit' to quit.")
      _ <- (for
        userInput <- Console.readLine
        _ <- ZIO.when(userInput.trim.toLowerCase != "exit") {
          for
            // Add user message to memory
            _ <- memory.add(ChatMessage(Role.User, userInput))
            
            // Get chat history
            history <- memory.get
            
            // Generate response
            response <- llm.completeChat(history)
            
            // Add assistant response to memory
            _ <- memory.add(response.message)
            
            // Display response
            _ <- Console.printLine(s"AI: ${response.message.content}")
          yield ()
        }
      ).repeatWhile(input => input.trim.toLowerCase != "exit")
    yield ()
    .provide(
      OpenAILLM.live,
      ChatMemory.volatile,
      ZLayer.succeed(OpenAIConfig(...))
    )
```

### Memory with Summarization

```scala
import zio.langchain.core.chain.Chain

// Chain that uses memory with summarization
def createChatChain(
  llm: LLM,
  memory: Memory
): Chain[Any, Throwable, String, String] = new Chain[Any, Throwable, String, String]:
  override def run(input: String): ZIO[Any, Throwable, String] =
    for
      // Add user message
      _ <- memory.add(ChatMessage(Role.User, input))
      
      // Get history
      history <- memory.get
      
      // Generate response
      response <- llm.completeChat(history)
        .mapError(e => e: Throwable)
      
      // Store response
      _ <- memory.add(response.message)
    yield response.message.content
```

## Building Custom Memory

You can create custom memory implementations for specific use cases:

```scala
import zio.langchain.core.memory.Memory
import zio.json.*

// Memory that persists to a file
class FileBackedMemory(filePath: Path) extends Memory:
  private val lock = new Object()
  
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    ZIO.attemptBlockingIO {
      lock.synchronized {
        // Read existing messages
        val existingMessages = if (Files.exists(filePath)) {
          Files.readString(filePath).fromJson[Seq[ChatMessage]].getOrElse(Seq.empty)
        } else {
          Seq.empty[ChatMessage]
        }
        
        // Add new message
        val updatedMessages = existingMessages :+ message
        
        // Write back to file
        Files.writeString(
          filePath, 
          updatedMessages.toJson,
          StandardOpenOption.CREATE, 
          StandardOpenOption.TRUNCATE_EXISTING
        )
      }
    }.mapError(e => MemoryError(e))
    
  // Implement other methods following similar pattern
```

## Best Practices

1. **Token Management**: Monitor and limit the number of tokens in memory to avoid exceeding LLM context limits.

2. **Message Filtering**: Consider filtering out irrelevant messages:
   ```scala
   def filterMessages(messages: Seq[ChatMessage]): Seq[ChatMessage] =
     messages.filter(msg => !isIrrelevant(msg.content))
   ```

3. **Selective Retrieval**: Not all history may be relevant for every query:
   ```scala
   def retrieveRelevant(query: String, messages: Seq[ChatMessage]): Seq[ChatMessage] =
     // Implementation that selects relevant messages
   ```

4. **Composite Memory**: Combine multiple memory strategies:
   ```scala
   class CompositeMemory(
     shortTermMemory: Memory,
     longTermMemory: Memory
   ) extends Memory
   ```

5. **Memory Compression**: Periodically compress memory to save tokens:
   ```scala
   def compressMemory(messages: Seq[ChatMessage], llm: LLM): ZIO[Any, Throwable, String] =
     // Implementation that summarizes messages
   ```

6. **Contextual Segmentation**: Organize memory into different contexts:
   ```scala
   class SegmentedMemory(
     segments: Map[String, Memory]
   ) extends Memory
   ```

7. **Security Considerations**:
   - Never store sensitive information in plain text
   - Consider encryption for persistent memory
   - Implement proper data retention policies
   - Allow users to clear their conversation history