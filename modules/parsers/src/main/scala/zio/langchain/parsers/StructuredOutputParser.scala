package zio.langchain.parsers

import zio.*
import zio.json.*
import zio.langchain.core.errors.{LLMError, OutputParsingError}
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.{ChatMessage, Role}

/**
 * StructuredOutputParser provides a high-level interface for generating structured outputs from LLMs.
 * It combines prompting strategies with output parsing to guide the model to produce outputs in the expected format.
 *
 * @tparam T The type to parse the output into
 */
class StructuredOutputParser[T] private (
  parser: OutputParser[T],
  promptTemplate: String => String
):
  /**
   * Generate a structured output using the provided LLM.
   *
   * @param prompt The user prompt
   * @param llm The LLM to use for generation
   * @return A ZIO effect that produces the parsed structured output or fails with an error
   */
  def generateStructured(
    prompt: String,
    llm: LLM
  ): ZIO[Any, LLMError | OutputParsingError, T] =
    val formattedPrompt = promptTemplate(prompt)
    
    for
      response <- llm.complete(formattedPrompt)
      parsed <- parser.parse(response)
        .mapError(e => e) // Preserve the OutputParsingError
    yield parsed

  /**
   * Generate a structured output using the provided LLM with retry capability.
   *
   * @param prompt The user prompt
   * @param llm The LLM to use for generation
   * @param maxRetries Maximum number of retry attempts
   * @return A ZIO effect that produces the parsed structured output or fails with an error
   */
  def generateStructuredWithRetry(
    prompt: String,
    llm: LLM,
    maxRetries: Int = 3
  ): ZIO[Any, LLMError | OutputParsingError, T] =
    val formattedPrompt = promptTemplate(prompt)
    
    for
      response <- llm.complete(formattedPrompt)
      parsed <- parser.parseWithRetry(response, llm, maxRetries)
    yield parsed

  /**
   * Generate a structured output using the provided LLM in a chat context.
   *
   * @param messages The chat messages
   * @param llm The LLM to use for generation
   * @return A ZIO effect that produces the parsed structured output or fails with an error
   */
  def generateStructuredChat(
    messages: Seq[ChatMessage],
    llm: LLM
  ): ZIO[Any, LLMError | OutputParsingError, T] =
    // Add format instructions to the last user message or create a new system message
    val enhancedMessages = addFormatInstructions(messages)
    
    for
      response <- llm.completeChat(enhancedMessages).map(_.message.contentAsString)
      parsed <- parser.parse(response)
        .mapError(e => e) // Preserve the OutputParsingError
    yield parsed

  /**
   * Add format instructions to the chat messages.
   *
   * @param messages The original chat messages
   * @return The enhanced messages with format instructions
   */
  private def addFormatInstructions(messages: Seq[ChatMessage]): Seq[ChatMessage] =
    val formatInstructions = parser.getFormatInstructions
    
    if messages.isEmpty then
      Seq(ChatMessage.system(s"You must respond in the following format:\n$formatInstructions"))
    else
      val lastUserMessageIndex = messages.lastIndexWhere(_.role == Role.User)
      
      if lastUserMessageIndex >= 0 then
        // Add format instructions to the last user message
        val lastUserMessage = messages(lastUserMessageIndex)
        val enhancedContent = s"${lastUserMessage.contentAsString}\n\nRespond in the following format:\n$formatInstructions"
        val enhancedMessage = lastUserMessage.withContent(enhancedContent)
        
        messages.updated(lastUserMessageIndex, enhancedMessage)
      else
        // Add a system message with format instructions
        messages :+ ChatMessage.system(s"Respond in the following format:\n$formatInstructions")

/**
 * Companion object for StructuredOutputParser.
 */
object StructuredOutputParser:
  /**
   * Creates a new StructuredOutputParser with the given parser and prompt template.
   *
   * @param parser The output parser to use
   * @param promptTemplate A function that formats the user prompt with format instructions
   * @return A new StructuredOutputParser
   */
  def apply[T](
    parser: OutputParser[T],
    promptTemplate: String => String = null
  ): StructuredOutputParser[T] =
    val actualTemplate = if (promptTemplate == null)
      (prompt: String) => defaultPromptTemplate(prompt, parser)
    else
      promptTemplate
    new StructuredOutputParser[T](parser, actualTemplate)

  /**
   * Creates a new StructuredOutputParser for JSON outputs.
   *
   * @param promptTemplate A function that formats the user prompt with format instructions
   * @return A new StructuredOutputParser for the specified type
   */
  def forJson[T: JsonDecoder: JsonEncoder: scala.reflect.ClassTag](
    promptTemplate: String => String = null
  ): StructuredOutputParser[T] =
    val parser = OutputParser.json[T]()
    val actualTemplate = if (promptTemplate == null)
      (prompt: String) => defaultPromptTemplate(prompt, parser)
    else
      promptTemplate
    new StructuredOutputParser[T](parser, actualTemplate)

  /**
   * Creates a new StructuredOutputParser with JSON schema validation.
   *
   * @param schema The JSON schema to validate against
   * @param promptTemplate A function that formats the user prompt with format instructions
   * @return A new StructuredOutputParser for the specified type
   */
  def withJsonSchema[T: JsonDecoder: JsonEncoder: scala.reflect.ClassTag](
    schema: JsonSchema,
    promptTemplate: String => String = null
  ): StructuredOutputParser[T] =
    val parser = JsonSchemaOutputParser[T](schema)
    val actualTemplate = if (promptTemplate == null)
      (prompt: String) => defaultPromptTemplate(prompt, parser)
    else
      promptTemplate
    new StructuredOutputParser[T](parser, actualTemplate)

  /**
   * Default prompt template that adds format instructions to the user prompt.
   *
   * @param prompt The user prompt
   * @param parser The output parser to get format instructions from
   * @return The formatted prompt
   */
  private def defaultPromptTemplate(prompt: String, parser: OutputParser[?]): String =
    s"""$prompt
       |
       |You must respond in the following format:
       |${parser.getFormatInstructions}
       |""".stripMargin

  /**
   * Implicit class to add structured output capabilities to LLM.
   */
  extension (llm: LLM)
    /**
     * Generate a structured output using the provided parser.
     *
     * @param prompt The user prompt
     * @param parser The output parser to use
     * @return A ZIO effect that produces the parsed structured output or fails with an error
     */
    def completeStructured[T](
      prompt: String,
      parser: OutputParser[T]
    ): ZIO[Any, LLMError | OutputParsingError, T] =
      val formattedPrompt = s"""$prompt
                               |
                               |You must respond in the following format:
                               |${parser.getFormatInstructions}
                               |""".stripMargin
      
      for
        response <- llm.complete(formattedPrompt)
        parsed <- parser.parse(response)
          .mapError(e => e) // Preserve the OutputParsingError
      yield parsed

    /**
     * Generate a structured output in a chat context using the provided parser.
     *
     * @param messages The chat messages
     * @param parser The output parser to use
     * @return A ZIO effect that produces the parsed structured output or fails with an error
     */
    def completeChatStructured[T](
      messages: Seq[ChatMessage],
      parser: OutputParser[T]
    ): ZIO[Any, LLMError | OutputParsingError, T] =
      // Add format instructions
      val formatInstructions = parser.getFormatInstructions
      val enhancedMessages = if messages.isEmpty then
        Seq(ChatMessage.system(s"You must respond in the following format:\n$formatInstructions"))
      else
        val lastUserMessageIndex = messages.lastIndexWhere(_.role == Role.User)
        
        if lastUserMessageIndex >= 0 then
          // Add format instructions to the last user message
          val lastUserMessage = messages(lastUserMessageIndex)
          val enhancedContent = s"${lastUserMessage.contentAsString}\n\nRespond in the following format:\n$formatInstructions"
          val enhancedMessage = lastUserMessage.withContent(enhancedContent)
          
          messages.updated(lastUserMessageIndex, enhancedMessage)
        else
          // Add a system message with format instructions
          messages :+ ChatMessage.system(s"Respond in the following format:\n$formatInstructions")
      
      for
        response <- llm.completeChat(enhancedMessages).map(_.message.contentAsString)
        parsed <- parser.parse(response)
          .mapError(e => e) // Preserve the OutputParsingError
      yield parsed