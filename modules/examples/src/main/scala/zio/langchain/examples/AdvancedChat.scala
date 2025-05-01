package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.LLM
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.*
import zio.langchain.memory.BufferMemory

/**
 * An advanced chat example that demonstrates the use of memory to maintain conversation history.
 * This example builds on the SimplifiedChat example but adds conversational memory.
 */
object AdvancedChat extends ZIOAppDefault:
  /**
   * The main program.
   * Sets up the OpenAI LLM, BufferMemory, and runs an enhanced chat interface.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Advanced Chat!")
      _ <- printLine("This example demonstrates conversational memory with chat history.")
      _ <- printLine("Type 'exit' to quit, 'clear' to reset conversation history.")
      _ <- printLine("")
      
      // Get the required services
      llm <- ZIO.service[LLM]
      memory <- ZIO.service[Memory]
      
      // Set up initial system message for the assistant
      _ <- memory.add(ChatMessage.system(
        "You are a helpful assistant that remembers previous parts of the conversation. " +
        "Provide concise and informative responses."
      ))
      
      // Run the chat loop
      _ <- chatLoop(llm, memory)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // HTTP Client dependency required by all API integrations
      Client.default,
      
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
      
      // BufferMemory with maximum history
      BufferMemory.layer(Some(20))
    ).catchAllCause { cause =>
      val errorMessage = cause.failureOption.fold("Unknown error")(_.getMessage)
      printLine(s"Error: $errorMessage")
    }
  
  /**
   * The advanced chat loop with memory support.
   * It maintains conversation history and allows clearing the history.
   *
   * @param llm The LLM service
   * @param memory The Memory service to store conversation history
   * @return A ZIO effect that completes when the user exits the chat
   */
  private def chatLoop(llm: LLM, memory: Memory): ZIO[Any, Throwable, Unit] = {
    for {
      // Prompt the user for input
      _ <- printLine(">")
      input <- readLine
      
      // Process the user input
      _ <- input.toLowerCase match {
        case "exit" => 
          printLine("Goodbye!")
        
        case "clear" => 
          for {
            _ <- memory.clear
            _ <- memory.add(ChatMessage.system(
              "You are a helpful assistant that remembers previous parts of the conversation. " +
              "Provide concise and informative responses."
            ))
            _ <- printLine("Conversation history cleared.")
            _ <- chatLoop(llm, memory)
          } yield ()
        
        case _ => 
          for {
            // Add user input to memory
            _ <- memory.add(ChatMessage.user(input))
            
            // Get full conversation history
            history <- memory.get
            
            // Send the full history to the LLM
            response <- llm.completeChat(history)
            
            // Add the assistant's response to memory
            _ <- memory.add(response.message)
            
            // Display the response
            _ <- printLine(response.message.contentAsString)
            _ <- printLine("")
            
            // Continue the chat loop
            _ <- chatLoop(llm, memory)
          } yield ()
      }
    } yield ()
  }