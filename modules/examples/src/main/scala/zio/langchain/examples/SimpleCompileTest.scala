package zio.langchain.examples

import zio.*
import zio.Console.*

import zio.langchain.core.model.LLM
import zio.langchain.integrations.openai.{OpenAIConfig, OpenAILLM}

/**
 * A simple test to verify that the ZIO LangChain core functionality compiles and works correctly.
 */
object SimpleCompileTest extends ZIOAppDefault:
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Simple Compile Test!")
      _ <- printLine("")
      
      // Get the LLM service
      llm <- ZIO.service[LLM]
      
      // Run a simple completion
      _ <- printLine("Running simple completion...")
      prompt = "Say hello to the world in one sentence."
      response <- llm.complete(prompt)
      _ <- printLine(s"LLM Response: $response")
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = sys.env.getOrElse("OPENAI_MODEL", "gpt-3.5-turbo"),
          temperature = 0.7,
          maxTokens = Some(100)
        )
      )
    )