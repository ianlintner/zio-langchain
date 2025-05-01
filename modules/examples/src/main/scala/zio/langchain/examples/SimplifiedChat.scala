package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

/**
 * A simplified chat application example using ZIO LangChain.
 * This provides a basic chatbot implementation with minimal dependencies.
 */
object SimplifiedChat extends ZIOAppDefault:
  /**
   * The main program.
   * Sets up the OpenAI LLM and runs a simple chat interface.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Simplified Chat!")
      _ <- printLine("Type 'exit' to quit.")
      _ <- printLine("")
      
      // Get the LLM service
      llm <- ZIO.service[LLM]
      
      // Run the chat loop
      _ <- chatLoop(llm)
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
      )
    )
  
  /**
   * The simplified chat loop.
   * It repeatedly prompts the user for input and sends it to the LLM.
   *
   * @param llm The LLM service
   * @return A ZIO effect that completes when the user exits the chat
   */
  private def chatLoop(llm: LLM): ZIO[Any, Throwable, Unit] = {
    for {
      // Prompt the user for input
      _ <- printLine(">")
      input <- readLine
      
      // Check if the user wants to exit
      _ <- if (input.toLowerCase == "exit") 
        printLine("Goodbye!")
      else 
        for {
          // Get response from LLM
          response <- llm.complete(input)
          // Display the response
          _ <- printLine(response)
          _ <- printLine("")
          // Continue the chat loop
          _ <- chatLoop(llm)
        } yield ()
    } yield ()
  }