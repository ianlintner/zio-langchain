package zio.langchain.core.agent

import zio.*
import zio.langchain.core.LangChainError
import zio.langchain.core.tool.Tool

/**
 * Interface for agents.
 * Agents are autonomous systems that can make decisions, use tools, and solve tasks.
 */
trait Agent:
  /**
   * Runs the agent with the given input.
   * The agent will process the input, make decisions, potentially use tools, and produce a final output.
   *
   * @param input The input to the agent, typically a task description or question
   * @return A ZIO effect that produces a string output, or fails with a LangChainError
   */
  def run(input: String): IO[LangChainError, String]

object Agent:
  /**
   * Configuration for agents.
   */
  trait Config:
    /**
     * The maximum number of iterations the agent should perform before giving up.
     * This prevents infinite loops in agent reasoning.
     *
     * @return The maximum number of iterations
     */
    def maxIterations: Int
    
    /**
     * The maximum number of tokens to generate in each agent step.
     * This helps control the verbosity and cost of agent operations.
     *
     * @return The maximum number of tokens
     */
    def maxTokens: Option[Int]
    
    /**
     * The temperature parameter for the LLM used by the agent.
     * Higher values make the output more random, lower values make it more deterministic.
     *
     * @return The temperature value
     */
    def temperature: Double

  /**
   * Default implementation of Config.
   *
   * @param maxIterations The maximum number of iterations
   * @param maxTokens The maximum number of tokens
   * @param temperature The temperature parameter
   */
  case class DefaultConfig(
    override val maxIterations: Int = 10,
    override val maxTokens: Option[Int] = None,
    override val temperature: Double = 0.7
  ) extends Config
  
  /**
   * Runs the agent with the given input.
   *
   * @param input The input to the agent
   * @return A ZIO effect that produces a string output, or fails with a LangChainError
   */
  def run(input: String): ZIO[Agent, LangChainError, String] =
    ZIO.service[Agent].flatMap(_.run(input))