package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.integrations.anthropic.*

/**
 * Example demonstrating the use of Anthropic Claude integration.
 *
 * To run this example:
 * 1. Set your Anthropic API key in the environment variable ANTHROPIC_API_KEY
 * 2. Run the example using: `sbt "examples/runMain zio.langchain.examples.AnthropicExample"`
 *
 * Alternatively, you can configure the Anthropic integration using HOCON configuration:
 * 1. Create an application.conf file with your Anthropic configuration
 * 2. Run the example
 */
object AnthropicExample extends ZIOAppDefault:

  val program = for
    // Get the LLM service
    llm <- ZIO.service[LLM]
    
    // Simple text completion
    _ <- printLine("Simple text completion:")
    completion <- llm.complete("Explain quantum computing in simple terms.")
    _ <- printLine(completion)
    _ <- printLine("")
    
    // Chat completion with system and user messages
    _ <- printLine("Chat completion with system and user messages:")
    messages = Seq(
      ChatMessage.system("You are a helpful assistant that provides concise answers."),
      ChatMessage.user("What are the main features of ZIO?")
    )
    chatResponse <- llm.completeChat(messages)
    _ <- printLine(chatResponse.message.contentAsString)
    _ <- printLine("")
    
    // Display token usage
    _ <- printLine("Token usage:")
    _ <- printLine(s"Prompt tokens: ${chatResponse.usage.promptTokens}")
    _ <- printLine(s"Completion tokens: ${chatResponse.usage.completionTokens}")
    _ <- printLine(s"Total tokens: ${chatResponse.usage.totalTokens}")
    
  yield ()

  override def run = program.provide(
    // Anthropic configuration layer
    ZLayer.succeed(
      AnthropicConfig(
        apiKey = sys.env.getOrElse("ANTHROPIC_API_KEY", ""),
        model = "claude-3-sonnet-20240229"
      )
    ),
    
    // Anthropic LLM layer - includes Client.default internally
    AnthropicLLM.live
  )