package zio.langchain.core.tool

import zio.*

import zio.langchain.core.errors.*

/**
 * Interface for tools that can be used by agents.
 * Tools provide a way for agents to interact with external systems or perform specific actions.
 *
 * @tparam R The environment type required by this tool
 * @tparam E The error type that can be produced by this tool
 */
trait Tool[-R, +E <: LangChainError]:
  /**
   * The name of the tool.
   * This should be a short, descriptive identifier that can be used by agents to reference the tool.
   *
   * @return The name of the tool
   */
  def name: String
  
  /**
   * A description of what the tool does, how to use it, and what inputs it expects.
   * This description is used by agents to understand when and how to use the tool.
   *
   * @return The description of the tool
   */
  def description: String
  
  /**
   * Executes the tool with the given input.
   *
   * @param input The input to the tool, typically a string
   * @return A ZIO effect that requires an environment R, produces a string output, or fails with an error E
   */
  def execute(input: String): ZIO[R, E, String]

/**
 * Companion object for Tool.
 */
object Tool:
  /**
   * Creates a new tool from a function.
   *
   * @param toolName The name of the tool
   * @param toolDescription The description of the tool
   * @param f The function to create a tool from
   * @tparam R The environment type required by the function
   * @tparam E The error type that can be produced by the function
   * @return A new tool that wraps the function
   */
  def apply[R, E <: LangChainError](
    toolName: String,
    toolDescription: String
  )(f: String => ZIO[R, E, String]): Tool[R, E] =
    new Tool[R, E]:
      override def name: String = toolName
      override def description: String = toolDescription
      override def execute(input: String): ZIO[R, E, String] = f(input)
  
  /**
   * Creates a ZIO accessor for a specific tool by name.
   *
   * @param toolName The name of the tool to access
   * @return A ZIO effect that requires a Map of tools and produces the tool with the given name, or fails if the tool is not found
   */
  def get(toolName: String): ZIO[Map[String, Tool[Any, LangChainError]], ToolExecutionError, Tool[Any, LangChainError]] =
    ZIO.serviceWithZIO[Map[String, Tool[Any, LangChainError]]] { tools =>
      ZIO.fromOption(tools.get(toolName))
        .mapError(_ => ToolExecutionError(s"Tool not found: $toolName", new NoSuchElementException()))
    }
  
  /**
   * Executes a specific tool by name with the given input.
   *
   * @param toolName The name of the tool to execute
   * @param input The input to the tool
   * @return A ZIO effect that requires a Map of tools and produces the output of the tool, or fails if the tool is not found or execution fails
   */
  def execute(toolName: String, input: String): ZIO[Map[String, Tool[Any, LangChainError]], LangChainError, String] =
    get(toolName).flatMap(_.execute(input))