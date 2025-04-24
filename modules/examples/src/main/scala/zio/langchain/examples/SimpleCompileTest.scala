package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.ExitCode

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.{OpenAIConfig, OpenAILLM}
import zio.langchain.chains.LLMChain

/**
 * A simple test to verify that the ZIO LangChain core functionality compiles and works correctly.
 * This example focuses only on the core modules that are known to compile successfully:
 * - Core domain models
 * - OpenAI integration
 * - LLMChain
 */
object SimpleCompileTest extends ZIOAppDefault:

  override def run =
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI configuration layer with validation
      ZLayer.fromZIO(
        ZIO.attempt {
          OpenAIConfig(
            apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
            model = sys.env.getOrElse("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature = 0.7,
            maxTokens = Some(150)
          )
        }.flatMap(config => 
          if (config.apiKey.trim.isEmpty) 
            ZIO.fail(new RuntimeException("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."))
          else 
            ZIO.succeed(config)
        )
      )
    ).catchAllCause { cause =>
      val error = cause.failureOption.getOrElse(new RuntimeException("Unknown error"))
      val message = error.getMessage
      val errorPrefix = if (message.contains("Authentication error")) {
        "Authentication Error"
      } else if (message.contains("Rate limit exceeded")) {
        "Rate Limit Error"
      } else if (message.contains("OpenAI server error")) {
        "OpenAI Server Error"
      } else if (message.contains("Request timed out")) {
        "Timeout Error"
      } else {
        "Unexpected Error"
      }
      
      for {
        _ <- printLine(s"$errorPrefix: $message")
      } yield ExitCode.failure
    }
    
  val program = for
    // Print welcome message
    _ <- printLine("Welcome to ZIO LangChain Simple Compile Test!")
    _ <- printLine("This example demonstrates the core functionality that compiles correctly.")
    _ <- printLine("")
    
    // Get the LLM service
    llm <- ZIO.service[LLM]
    
    // PART 1: Direct LLM usage
    _ <- printLine("=== PART 1: Direct LLM Usage ===")
    
    // Example 1: Simple text completion
    _ <- printLine("1. Running simple text completion...")
    textPrompt = "Say hello to the world in one sentence."
    textResponse <- llm.complete(textPrompt)
    _ <- printLine(s"LLM Response: $textResponse")
    _ <- printLine("")
    
    // Example 2: Chat completion with multiple messages
    _ <- printLine("2. Running chat completion with multiple messages...")
    chatMessages = Seq(
      ChatMessage.system("You are a helpful assistant that provides concise responses."),
      ChatMessage.user("What are the main components of ZIO?")
    )
    chatResponse <- llm.completeChat(chatMessages)
    _ <- printLine(s"LLM Response: ${chatResponse.message.contentAsString}")
    _ <- printLine(s"Token Usage: ${chatResponse.usage.totalTokens} tokens")
    _ <- printLine("")
    
    // PART 2: LLMChain usage
    _ <- printLine("=== PART 2: LLMChain Usage ===")
    
    // Create a simple LLMChain for string input/output
    _ <- printLine("Creating and using a simple LLMChain...")
    promptTemplate = "Explain the following concept in simple terms: {input}"
    stringChain = LLMChain.string(promptTemplate, llm)
    
    // Use the chain
    chainInput = "functional programming"
    chainResponse <- stringChain.run(chainInput)
    _ <- printLine(s"Chain Response: $chainResponse")
    
    // Final message
    _ <- printLine("")
    _ <- printLine("All tests completed successfully!")
  yield ()