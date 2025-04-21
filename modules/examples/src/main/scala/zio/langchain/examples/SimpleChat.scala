package zio.langchain.examples

import zio.*
import zio.Console.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.memory.*
import zio.langchain.core.errors.{MemoryError, LangChainError}
import zio.langchain.memory.BufferMemory
import zio.langchain.integrations.openai.*

/**
 * A simple chat application example using ZIO LangChain.
 * This example demonstrates how to use the LLM and Memory components to create a chat application.
 */
object SimpleChat extends ZIOAppDefault:
  /**
   * The main program.
   * It sets up the OpenAI LLM and a volatile memory, then runs the chat loop.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Chat!")
      _ <- printLine("Type 'exit' to quit.")
      _ <- printLine("")
      
      // Get the LLM and Memory services
      llm <- ZIO.service[LLM]
      memory <- ZIO.service[Memory]
      
      // Add a system message to set the context
      _ <- memory.add(ChatMessage.system(
        "You are a helpful assistant. Be concise and clear in your responses."
      ))
      
      // Run the chat loop
      _ <- chatLoop(llm, memory)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo",
          temperature = 0.7
        )
      ),
      // Volatile memory layer
      ZLayer.succeed {
        new Memory {
          private val messages = scala.collection.mutable.ArrayBuffer[ChatMessage]()
          
          override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
            ZIO.succeed {
              messages.append(message)
              ()
            }
          
          override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
            ZIO.succeed(messages.toSeq)
          
          override def clear: ZIO[Any, MemoryError, Unit] =
            ZIO.succeed {
              messages.clear()
              ()
            }
        }
      }
    )
  
  /**
   * The chat loop.
   * It repeatedly prompts the user for input, sends it to the LLM, and displays the response.
   *
   * @param llm The LLM service
   * @param memory The Memory service
   * @return A ZIO effect that completes when the user exits the chat
   */
  private def chatLoop(llm: LLM, memory: Memory): ZIO[Any, Throwable, Unit] =
    for
      // Prompt the user for input
      _ <- printLine(">")
      input <- readLine
      
      // Check if the user wants to exit
      result <- if (input.toLowerCase == "exit") then
        printLine("Goodbye!")
      else
        // Process the user input
        processUserInput(input, llm, memory)
    yield result
  
  /**
   * Process user input and continue the chat loop.
   */
  private def processUserInput(input: String, llm: LLM, memory: Memory): ZIO[Any, Throwable, Unit] =
    for 
      // Add the user message to memory
      _ <- memory.add(ChatMessage.user(input))
      
      // Get all messages from memory
      messages <- memory.get
      
      // Send the messages to the LLM
      response <- llm.completeChat(messages)
      
      // Add the assistant's response to memory
      _ <- memory.add(response.message)
      
      // Display the response
      _ <- printLine(response.message.contentAsString)
      _ <- printLine("")
      
      // Continue the chat loop
      result <- chatLoop(llm, memory)
    yield result