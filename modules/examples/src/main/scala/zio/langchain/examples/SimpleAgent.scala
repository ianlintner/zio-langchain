package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.agent.Agent
import zio.langchain.core.tool.Tool
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.{OpenAIConfig, OpenAILLM}

/**
 * A simple agent example using ZIO LangChain.
 * 
 * This example demonstrates the use of a basic agent that can use tools to accomplish tasks.
 * 
 * To run this example:
 * 1. Set your OpenAI API key in the environment variable OPENAI_API_KEY
 * 2. Run the example using: `sbt "examples/runMain zio.langchain.examples.SimpleAgent"`
 */
object SimpleAgent extends ZIOAppDefault:

  // Define a simple tool that can multiply two numbers
  val calculatorTool: Tool[Any, LangChainError] = Tool(
    toolName = "calculator",
    toolDescription = "A calculator that can multiply two numbers. Input should be two numbers separated by a comma."
  ) { input =>
    // Parse input and multiply numbers
    val parts = input.split(",").map(_.trim)
    if (parts.length != 2) {
      ZIO.fail(ToolExecutionError("Calculator tool expects two numbers separated by a comma", new IllegalArgumentException()))
    } else {
      ZIO.attempt {
        val num1 = parts(0).toDouble
        val num2 = parts(1).toDouble
        val result = num1 * num2
        s"The result of multiplying $num1 and $num2 is $result"
      }.mapError(e => ToolExecutionError(s"Failed to perform calculation: ${e.getMessage}", e))
    }
  }

  // Define a simple agent implementation
  val simpleAgent: RIO[LLM, Agent] = ZIO.service[LLM].map { llm =>
    new Agent:
      override def run(input: String): IO[LangChainError, String] = {
        val systemPrompt = """You are a helpful assistant that can use tools to solve problems.
          |When you need to use a tool, format your response like this:
          |TOOL: tool_name
          |INPUT: tool_input
          |
          |After receiving the tool's result, incorporate it into your final answer.
          |Available tools:
          |calculator - A calculator that can multiply two numbers. Input should be two numbers separated by a comma.
          |""".stripMargin
          
        // Define messages outside the for-comprehension
        val initialMessages = Seq(
          ChatMessage.system(systemPrompt),
          ChatMessage.user(input)
        )
        
        // Now use a properly structured for-comprehension
        for {
          // Ask the LLM for an initial plan
          initialResponse <- llm.completeChat(initialMessages)
          content = initialResponse.message.contentAsString
          
          // Parse the response to see if it wants to use a tool
          result <- if (content.contains("TOOL: calculator")) {
            // Extract the tool input
            val pattern = "INPUT: (.+)".r
            val toolInput = pattern.findFirstMatchIn(content).map(_.group(1)).getOrElse("")
            
            // Use the tool
            calculatorTool.execute(toolInput).flatMap { toolResult =>
              // Ask LLM to incorporate the tool result
              val followupMessages = initialMessages ++ Seq(
                ChatMessage.assistant(content),
                ChatMessage.system(s"Tool result: $toolResult")
              )
              llm.completeChat(followupMessages).map(_.message.contentAsString)
            }
          } else {
            // No tool use needed, return the response directly
            ZIO.succeed(content)
          }
        } yield result
      }
  }

  // Main program
  val program = for {
    // Print welcome message
    _ <- printLine("Welcome to ZIO LangChain Simple Agent Example!")
    _ <- printLine("")
    
    // Get the agent service
    agent <- simpleAgent
    
    // Run agent with a simple query
    _ <- printLine("Running agent with query: 'What is 123 multiplied by 456?'")
    result <- agent.run("What is 123 multiplied by 456?")
    _ <- printLine("Agent result:")
    _ <- printLine(result)
    _ <- printLine("")
    
    // Try another query
    _ <- printLine("Running agent with query: 'Can you help me calculate 7.5 times 12?'")
    result2 <- agent.run("Can you help me calculate 7.5 times 12?")
    _ <- printLine("Agent result:")
    _ <- printLine(result2)
  } yield ()

  override def run = program.provide(
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
  ).catchAllCause { cause =>
    val error = cause.failureOption.getOrElse(new RuntimeException("Unknown error"))
    printLine(s"Agent execution error: ${error.getMessage}")
      .as(ExitCode.failure)
  }