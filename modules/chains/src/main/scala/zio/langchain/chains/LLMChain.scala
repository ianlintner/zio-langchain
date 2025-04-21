package zio.langchain.chains

import zio.*
import zio.stream.ZStream

import zio.langchain.core.model.LLM
import zio.langchain.core.chain.Chain
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*

/**
 * A Chain that uses an LLM to generate responses based on a prompt template.
 * This is one of the simplest and most commonly used chains, serving as a foundation for more complex chains.
 *
 * @param promptTemplate The template string with placeholders (e.g., "{input}")
 * @param llm The LLM to use for generating responses
 * @param memory Optional memory to maintain conversation history
 * @param outputParser Optional function to parse the LLM output
 */
class LLMChain[I, O] private (
  promptTemplate: String,
  llm: LLM,
  memory: Option[Memory] = None,
  outputParser: Option[String => O] = None
) extends Chain[Any, LangChainError, I, O]:
  /**
   * Runs the chain with the given input.
   *
   * @param input The input to the chain
   * @return A ZIO effect that produces an output O or fails with a LangChainError
   */
  override def run(input: I): ZIO[Any, LangChainError, O] =
    for
      // Convert input to string representation for template substitution
      inputStr = input.toString
      
      // Apply the input to the prompt template
      filledPrompt = promptTemplate.replace("{input}", inputStr)
      
      // Retrieve conversation history if memory is available
      previousMessages <- memory.map(_.get).getOrElse(ZIO.succeed(Seq.empty))
      
      // Create the message sequence (history + new user message)
      messages = previousMessages :+ ChatMessage.user(filledPrompt)
      
      // Get response from the LLM
      response <- llm.completeChat(messages)
      
      // Store the interaction in memory if available
      _ <- memory.map(_.add(response.message)).getOrElse(ZIO.unit)
      
      // Parse the output if a parser is provided
      result = outputParser match
        case Some(parser) => parser(response.message.contentAsString)
        case None => 
          // If no parser, try to cast the content to type O (will throw if incompatible)
          response.message.contentAsString.asInstanceOf[O]
    yield result

/**
 * Companion object for LLMChain.
 */
object LLMChain:
  /**
   * Creates an LLMChain that handles string input and output.
   *
   * @param promptTemplate The template string with placeholders (e.g., "{input}")
   * @param llm The LLM to use for generating responses
   * @param memory Optional memory to maintain conversation history
   * @return An LLMChain instance for string processing
   */
  def string(
    promptTemplate: String,
    llm: LLM,
    memory: Option[Memory] = None
  ): LLMChain[String, String] =
    new LLMChain[String, String](promptTemplate, llm, memory, None)
  
  /**
   * Creates an LLMChain with custom input and output types.
   *
   * @param promptTemplate The template string with placeholders (e.g., "{input}")
   * @param llm The LLM to use for generating responses
   * @param parser A function to parse the LLM output to type O
   * @param memory Optional memory to maintain conversation history
   * @return An LLMChain instance with custom typing
   */
  def typed[I, O](
    promptTemplate: String,
    llm: LLM,
    parser: String => O,
    memory: Option[Memory] = None
  ): LLMChain[I, O] =
    new LLMChain[I, O](promptTemplate, llm, memory, Some(parser))
  
  /**
   * Creates a ZLayer that provides an LLMChain for string processing.
   *
   * @param promptTemplate The template string with placeholders
   * @return A ZLayer that requires an LLM and optionally a Memory, and provides an LLMChain
   */
  def stringLayer(
    promptTemplate: String
  ): ZLayer[LLM, Nothing, LLMChain[String, String]] =
    ZLayer {
      for
        llm <- ZIO.service[LLM]
        memoryOpt = None
      yield string(promptTemplate, llm, memoryOpt)
    }
  
  /**
   * Creates a ZLayer that provides an LLMChain with custom input and output types.
   *
   * @param promptTemplate The template string with placeholders
   * @param parser A function to parse the LLM output to type O
   * @return A ZLayer that requires an LLM and optionally a Memory, and provides an LLMChain
   */
  def typedLayer[I, O](
    promptTemplate: String,
    parser: String => O
  ): ZLayer[LLM, Nothing, LLMChain[I, O]] =
    ZLayer {
      for
        llm <- ZIO.service[LLM]
        memoryOpt = None
      yield typed[I, O](promptTemplate, llm, parser, memoryOpt)
    }