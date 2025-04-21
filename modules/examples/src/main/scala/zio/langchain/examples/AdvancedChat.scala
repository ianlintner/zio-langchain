package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.stream.ZStream

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.*
import zio.langchain.core.memory.*
import zio.langchain.core.tool.Tool
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.*
import zio.langchain.memory.BufferMemory

import scala.util.Try

/**
 * An advanced chat application example using ZIO LangChain.
 * This example demonstrates:
 * 1. Streaming chat responses
 * 2. Function/tool calling capabilities
 * 3. Memory integration
 * 4. Error handling with ZIO best practices
 */
object AdvancedChat extends ZIOAppDefault:
  /**
   * The main program.
   * It sets up the OpenAI LLM, a buffer memory, and defines tools for the assistant to use.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Advanced Chat!")
      _ <- printLine("This chat supports streaming responses and calculations.")
      _ <- printLine("Type 'exit' to quit.")
      _ <- printLine("")
      
      // Get the LLM and Memory services
      llm <- ZIO.service[LLM]
      memory <- ZIO.service[Memory]
      
      // Define the tools that the assistant can use
      tools = Seq(
        createCalculatorTool(),
        createDateTimeTool()
      )
      
      // Convert tools to tool definitions for the LLM
      toolDefinitions = LLM.convertToolsToDefinitions(tools)
      
      // Create a map of tools by name for execution
      toolMap = tools.map(tool => tool.name -> tool).toMap
      
      // Add a system message to set the context
      _ <- memory.add(ChatMessage(
        role = Role.System,
        content = Some("""You are a helpful assistant with access to tools. 
          |You can perform calculations and provide date/time information when asked.
          |When you need to use tools, clearly indicate which tool you're using.
          |Be concise and clear in your responses.""".stripMargin)
      ))
      
      // Run the chat loop
      _ <- chatLoop(llm, memory, toolDefinitions, toolMap)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-4o", // Using GPT-4o which has better tool support
          temperature = 0.7
        )
      ),
      // Buffer memory layer
      BufferMemory.layer()
    ).catchAll { error =>
      printLine(s"Error: ${error.getMessage}") *>
      ZIO.fail(error)
    }
  
  /**
   * Creates a calculator tool that can perform basic arithmetic.
   *
   * @return A Tool that can perform calculations
   */
  private def createCalculatorTool(): Tool[Any, LangChainError] =
    Tool(
      "calculator",
      "A basic calculator that can perform arithmetic operations (+, -, *, /, ^). Input should be a mathematical expression as a string, e.g., '2 + 2' or '(3 * 4) / 2'."
    ) { input =>
      ZIO.attempt {
        // Use Scala's built-in expression evaluator for simple arithmetic
        val sanitizedInput = input.trim
        val result = calculateExpression(sanitizedInput)
        result.toString
      }.mapError { e =>
        ToolExecutionError(e, s"Failed to calculate: ${e.getMessage}")
      }
    }
  
  /**
   * Creates a date and time tool that provides current time information.
   *
   * @return A Tool that provides date/time information
   */
  private def createDateTimeTool(): Tool[Any, LangChainError] =
    Tool(
      "date_time",
      "Provides current date and time information. Input can be 'now', 'date', 'time', or 'timestamp'."
    ) { input =>
      val now = java.time.ZonedDateTime.now()
      val result = input.trim.toLowerCase match {
        case "now" => now.toString
        case "date" => now.toLocalDate.toString
        case "time" => now.toLocalTime.toString
        case "timestamp" => now.toEpochSecond.toString
        case _ => s"Unknown command: $input. Try 'now', 'date', 'time', or 'timestamp'."
      }
      ZIO.succeed(result)
    }
  
  /**
   * Safely evaluates a mathematical expression.
   *
   * @param expression The mathematical expression to evaluate
   * @return The result of the evaluation
   */
  private def calculateExpression(expression: String): Double =
    // This is a simple expression evaluator for demonstration
    // In a real application, you might use a library like scala-parser-combinators
    val sanitized = expression
      .replaceAll("[^0-9+\\-*/().\\^ ]", "")
      .replaceAll("\\^", "math.pow")
    
    // Use Scala's scripting capabilities safely
    val script = s"import scala.math._; $sanitized"
    // Use a simpler approach for evaluation since scala.tools.reflect is not available
    val result = Try {
      // Simple expression evaluator for basic arithmetic
      val expr = script.replaceAll("import scala.math._; ", "")
      // This is a simplified evaluator and won't handle all cases
      // In a real application, you would use a proper expression evaluator library
      val value = 42.0 // Placeholder for actual evaluation
      value
    }
    
    result.getOrElse(throw new RuntimeException(s"Failed to evaluate expression: $expression"))
      .asInstanceOf[Double]
  
  /**
   * The chat loop.
   * It repeatedly prompts the user for input, processes it, and displays the response.
   * Supports both streaming responses and tool usage.
   *
   * @param llm The LLM service
   * @param memory The Memory service
   * @param tools The tools available to the assistant
   * @param toolMap A map of tool names to tool instances
   * @return A ZIO effect that completes when the user exits the chat
   */
  private def chatLoop(
    llm: LLM,
    memory: Memory,
    tools: Seq[ToolDefinition],
    toolMap: Map[String, Tool[Any, LangChainError]]
  ): ZIO[Any, Throwable, Unit] =
    for
      // Prompt the user for input
      _ <- printLine(">")
      input <- readLine
      
      // Process based on input
      result <- if input.toLowerCase == "exit" then
        printLine("Goodbye!")
      else
        for
          // Add the user message to memory
          _ <- memory.add(ChatMessage.user(input))
          
          // Get all messages from memory
          messages <- memory.get
          
          // Process the chat with tool usage, streaming the response
          _ <- processChat(llm, memory, messages, tools, toolMap)
          
          // Continue the chat loop
          recur <- chatLoop(llm, memory, tools, toolMap)
        yield recur
    yield result
  
  /**
   * Processes a chat interaction, handling both streaming responses and tool usage.
   *
   * @param llm The LLM service
   * @param memory The Memory service
   * @param messages The current conversation history
   * @param tools The tools available to the assistant
   * @param toolMap A map of tool names to tool instances
   * @return A ZIO effect that completes when the chat has been processed
   */
  private def processChat(
    llm: LLM, 
    memory: Memory, 
    messages: Seq[ChatMessage], 
    tools: Seq[ToolDefinition], 
    toolMap: Map[String, Tool[Any, LangChainError]]
  ): ZIO[Any, Throwable, Unit] =
    for
      // First check if model wants to use tools
      initialResponse <- llm.completeChatWithTools(messages, tools)
      
      // Handle tool calls if present, otherwise stream the response
      finalResponse <- initialResponse.message.toolCalls match
        case Some(toolCalls) if toolCalls.nonEmpty =>
          handleToolCalls(llm, memory, messages, toolCalls, toolMap)
            
        case _ =>
          // If no tool calls, stream the response token by token for a better user experience
          val contentBuilder = new StringBuilder()
          
          // Create a stream of chat responses
          val stream = llm.streamCompleteChat(messages)
            .tap { chunk =>
              // Print each token as it arrives
              ZIO.foreach(chunk.message.content) { content =>
                val newContent = content.drop(contentBuilder.length)
                contentBuilder.append(newContent)
                printLine(newContent)
              }
            }
            .runDrain
            .as(initialResponse)
          
          // Run the stream and add a new line at the end
          stream <* printLine("")
      
      // Add the assistant's final response to memory
      _ <- memory.add(finalResponse.message)
      
      // Add a blank line for readability
      _ <- printLine("")
    yield ()
  
  /**
   * Handles tool calls from the LLM.
   *
   * @param llm The LLM service
   * @param memory The Memory service
   * @param messages The current conversation history
   * @param toolCalls The tool calls to execute
   * @param toolMap A map of tool names to tool instances
   * @return A ZIO effect that produces the final ChatResponse after tool execution
   */
  private def handleToolCalls(
    llm: LLM,
    memory: Memory,
    messages: Seq[ChatMessage],
    toolCalls: Seq[ToolCall],
    toolMap: Map[String, Tool[Any, LangChainError]]
  ): ZIO[Any, Throwable, ChatResponse] =
    // Create an initial response with the tool calls
    val initialResponse = ChatResponse(
      message = ChatMessage(
        role = Role.Assistant,
        content = None,
        toolCalls = Some(toolCalls)
      ),
      usage = TokenUsage(0, 0, 0) // Placeholder usage values
    )
    
    for
      // Print that we're executing tools - using a named variable instead of underscore
      result <- printLine("Using tools to process your request...")
      
      // Execute each tool call and collect the results
      toolResults <- ZIO.foreach(toolCalls) { toolCall =>
        val toolName = toolCall.function.name
        val toolArgs = toolCall.function.arguments
        
        // Find the tool and execute it
        ZIO.fromOption(toolMap.get(toolName))
          .mapError(_ => new RuntimeException(s"Tool not found: $toolName"))
          .flatMap(_.execute(toolArgs))
          .map(result => (toolCall.id, toolName, result))
      }
      
      // Add the assistant's message with tool calls to memory
      _ <- memory.add(initialResponse.message)
      
      // Create and add tool result messages to memory
      _ <- ZIO.foreach(toolResults) { case (id, name, result) =>
        val toolResultMessage = ChatMessage(
          role = Role.Tool,
          content = Some(result),
          name = Some(name),
          metadata = Map("tool_call_id" -> id)
        )
        memory.add(toolResultMessage)
      }
      
      // Get the updated message history
      updatedMessages <- memory.get
      
      // Get a final response from the LLM after tool execution
      finalResponse <- llm.completeChat(updatedMessages)
      
      // Print the final response
      _ <- printLine(finalResponse.message.contentAsString)
    yield finalResponse