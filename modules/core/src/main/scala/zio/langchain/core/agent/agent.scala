package zio.langchain.core.agent

import zio.*

import zio.langchain.core.errors.*
import zio.langchain.core.tool.Tool

/**
 * Interface for agents.
 * Agents are autonomous systems that can make decisions, use tools, and solve tasks.
 *
 * @tparam R The environment type required by this agent
 * @tparam E The error type that can be produced by this agent
 */
trait Agent[-R, +E <: LangChainError]:
  /**
   * Runs the agent with the given input.
   * The agent will process the input, make decisions, potentially use tools, and produce a final output.
   *
   * @param input The input to the agent, typically a task description or question
   * @return A ZIO effect that requires an environment R, produces a string output, or fails with an error E
   */
  def run(input: String): ZIO[R, E, String]

/**
 * Companion object for Agent.
 */
object Agent:
  /**
   * Creates a ZIO accessor for the Agent service.
   *
   * @return A ZIO effect that requires an Agent and produces the Agent
   */
  def get[R, E <: LangChainError]: ZIO[Agent[R, E], Nothing, Agent[R, E]] = 
    ZIO.service[Agent[R, E]]
  
  /**
   * Runs the agent with the given input using the Agent service.
   *
   * @param input The input to the agent
   * @return A ZIO effect that requires an Agent and produces a string output, or fails with an error
   */
  def run[R, E <: LangChainError](input: String): ZIO[Agent[R, E], E, String] =
    ZIO.serviceWithZIO[Agent[R, E]](_.run(input))

/**
 * Configuration for agents.
 */
trait AgentConfig:
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
 * Default implementation of AgentConfig.
 *
 * @param maxIterations The maximum number of iterations
 * @param maxTokens The maximum number of tokens
 * @param temperature The temperature parameter
 */
case class DefaultAgentConfig(
  override val maxIterations: Int = 10,
  override val maxTokens: Option[Int] = None,
  override val temperature: Double = 0.7
) extends AgentConfig